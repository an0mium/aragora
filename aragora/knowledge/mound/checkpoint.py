"""
Knowledge Mound Checkpoint Store.

Provides state persistence and recovery for the Knowledge Mound:
- Snapshot current KM state (nodes, relationships, metadata)
- Restore KM from checkpoint
- Incremental checkpoints
- Checkpoint management (list, delete, compare)

Usage:
    from aragora.knowledge.mound.checkpoint import KMCheckpointStore

    # Initialize
    store = KMCheckpointStore(mound, checkpoint_dir="/path/to/checkpoints")

    # Create checkpoint
    checkpoint_id = await store.create_checkpoint(
        description="Before migration",
        include_vectors=True,
    )

    # Restore from checkpoint
    await store.restore_checkpoint(checkpoint_id)

    # List checkpoints
    checkpoints = await store.list_checkpoints()
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


@dataclass
class KMCheckpointMetadata:
    """Metadata for a Knowledge Mound checkpoint."""

    id: str
    created_at: str
    description: str
    workspace_id: str
    mound_version: str = "1.0"

    # Statistics
    node_count: int = 0
    relationship_count: int = 0
    workspace_count: int = 0

    # Size info
    compressed: bool = True
    size_bytes: int = 0
    checksum: str = ""

    # Optional inclusions
    includes_vectors: bool = False
    includes_culture: bool = True
    includes_staleness: bool = True

    # Incremental checkpoint info
    incremental: bool = False
    parent_checkpoint_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KMCheckpointContent:
    """Content of a Knowledge Mound checkpoint."""

    # Nodes as list of dicts
    nodes: List[Dict[str, Any]] = field(default_factory=list)

    # Relationships as list of dicts
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    # Culture patterns
    culture_patterns: Dict[str, Any] = field(default_factory=dict)

    # Staleness tracking state
    staleness_state: Dict[str, Any] = field(default_factory=dict)

    # Vector embeddings (optional, large)
    vector_embeddings: Dict[str, List[float]] = field(default_factory=dict)

    # Workspace metadata
    workspace_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "relationships": self.relationships,
            "culture_patterns": self.culture_patterns,
            "staleness_state": self.staleness_state,
            "vector_embeddings": self.vector_embeddings,
            "workspace_metadata": self.workspace_metadata,
        }


@dataclass
class RestoreResult:
    """Result of restoring from checkpoint."""

    success: bool
    checkpoint_id: str
    nodes_restored: int = 0
    relationships_restored: int = 0
    culture_restored: bool = False
    duration_ms: int = 0
    errors: List[str] = field(default_factory=list)


class KMCheckpointStore:
    """
    Checkpoint store for Knowledge Mound state persistence.

    Provides full and incremental checkpointing of KM state,
    enabling disaster recovery and migration.
    """

    VERSION = "1.0"

    def __init__(
        self,
        mound: "KnowledgeMound",
        checkpoint_dir: Optional[str] = None,
        compress: bool = True,
        max_checkpoints: int = 10,
    ):
        """
        Initialize checkpoint store.

        Args:
            mound: KnowledgeMound instance to checkpoint
            checkpoint_dir: Directory for checkpoint storage
            compress: Whether to gzip compress checkpoints
            max_checkpoints: Maximum checkpoints to keep (oldest are pruned)
        """
        self.mound = mound
        self.compress = compress
        self.max_checkpoints = max_checkpoints

        # Set checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            # Default to .aragora/km_checkpoints
            self.checkpoint_dir = Path.home() / ".aragora" / "km_checkpoints"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def create_checkpoint(
        self,
        description: str = "",
        include_vectors: bool = False,
        include_culture: bool = True,
        include_staleness: bool = True,
        incremental: bool = False,
        parent_checkpoint_id: Optional[str] = None,
    ) -> str:
        """
        Create a checkpoint of current KM state.

        Args:
            description: Human-readable description
            include_vectors: Include vector embeddings (large)
            include_culture: Include culture patterns
            include_staleness: Include staleness tracking state
            incremental: Create incremental checkpoint (only changes)
            parent_checkpoint_id: Parent checkpoint for incremental

        Returns:
            Checkpoint ID
        """
        start_time = time.time()

        # Generate checkpoint ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"km_ckpt_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        logger.info(f"Creating KM checkpoint: {checkpoint_id}")

        # Collect content
        content = KMCheckpointContent()

        # Export nodes
        nodes = await self._export_nodes(incremental, parent_checkpoint_id)
        content.nodes = nodes

        # Export relationships
        relationships = await self._export_relationships(incremental, parent_checkpoint_id)
        content.relationships = relationships

        # Export culture patterns
        if include_culture:
            content.culture_patterns = await self._export_culture()

        # Export staleness state
        if include_staleness:
            content.staleness_state = await self._export_staleness()

        # Export vector embeddings (optional - can be large)
        if include_vectors:
            content.vector_embeddings = await self._export_vectors()

        # Export workspace metadata
        content.workspace_metadata = {
            "workspace_id": self.mound.workspace_id,
            "config": {
                "backend": self.mound.config.backend.value if hasattr(self.mound.config.backend, "value") else str(self.mound.config.backend),
                "enable_staleness_detection": self.mound.config.enable_staleness_detection,
                "enable_culture_accumulator": self.mound.config.enable_culture_accumulator,
            },
        }

        # Serialize and optionally compress
        content_json = json.dumps(content.to_dict(), default=str)

        if self.compress:
            content_bytes = gzip.compress(content_json.encode("utf-8"))
            extension = ".json.gz"
        else:
            content_bytes = content_json.encode("utf-8")
            extension = ".json"

        # Calculate checksum
        checksum = hashlib.sha256(content_bytes).hexdigest()

        # Create metadata
        metadata = KMCheckpointMetadata(
            id=checkpoint_id,
            created_at=datetime.now().isoformat(),
            description=description,
            workspace_id=self.mound.workspace_id,
            mound_version=self.VERSION,
            node_count=len(content.nodes),
            relationship_count=len(content.relationships),
            workspace_count=1,
            compressed=self.compress,
            size_bytes=len(content_bytes),
            checksum=checksum,
            includes_vectors=include_vectors,
            includes_culture=include_culture,
            includes_staleness=include_staleness,
            incremental=incremental,
            parent_checkpoint_id=parent_checkpoint_id,
        )

        # Write checkpoint files
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Write metadata
        metadata_file = checkpoint_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata.to_dict(), indent=2))

        # Write content
        content_file = checkpoint_path / f"content{extension}"
        content_file.write_bytes(content_bytes)

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Created KM checkpoint {checkpoint_id}: "
            f"{len(content.nodes)} nodes, {len(content.relationships)} relationships, "
            f"{len(content_bytes)} bytes, {duration_ms}ms"
        )

        # Prune old checkpoints
        await self._prune_old_checkpoints()

        return checkpoint_id

    async def restore_checkpoint(
        self,
        checkpoint_id: str,
        clear_existing: bool = True,
        restore_vectors: bool = True,
    ) -> RestoreResult:
        """
        Restore KM state from checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to restore
            clear_existing: Clear existing KM data before restore
            restore_vectors: Restore vector embeddings if present

        Returns:
            RestoreResult with statistics
        """
        start_time = time.time()
        errors: List[str] = []

        logger.info(f"Restoring KM from checkpoint: {checkpoint_id}")

        # Load checkpoint
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        if not checkpoint_path.exists():
            return RestoreResult(
                success=False,
                checkpoint_id=checkpoint_id,
                errors=[f"Checkpoint not found: {checkpoint_id}"],
            )

        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        if not metadata_file.exists():
            return RestoreResult(
                success=False,
                checkpoint_id=checkpoint_id,
                errors=["Checkpoint metadata not found"],
            )

        metadata_dict = json.loads(metadata_file.read_text())
        metadata = KMCheckpointMetadata(**metadata_dict)

        # Load content
        if metadata.compressed:
            content_file = checkpoint_path / "content.json.gz"
        else:
            content_file = checkpoint_path / "content.json"

        # Verify checksum first (before decompression)
        content_bytes = content_file.read_bytes()
        actual_checksum = hashlib.sha256(content_bytes).hexdigest()
        if actual_checksum != metadata.checksum:
            errors.append(f"Checksum mismatch: expected {metadata.checksum}, got {actual_checksum}")

        # Decompress and parse content
        try:
            if metadata.compressed:
                content_json = gzip.decompress(content_bytes).decode("utf-8")
            else:
                content_json = content_bytes.decode("utf-8")
            content_dict = json.loads(content_json)
        except (gzip.BadGzipFile, OSError) as e:
            return RestoreResult(
                success=False,
                checkpoint_id=checkpoint_id,
                errors=errors + [f"Failed to decompress checkpoint: {e}"],
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return RestoreResult(
                success=False,
                checkpoint_id=checkpoint_id,
                errors=errors + [f"Failed to parse checkpoint content: {e}"],
            )

        # Clear existing data if requested
        if clear_existing:
            await self._clear_mound_data()

        # Restore nodes
        nodes_restored = 0
        for node_dict in content_dict.get("nodes", []):
            try:
                await self._restore_node(node_dict)
                nodes_restored += 1
            except (KeyError, TypeError, ValueError) as e:
                errors.append(f"Failed to restore node {node_dict.get('id')} (data error): {e}")
            except (OSError, ConnectionError, RuntimeError) as e:
                errors.append(f"Failed to restore node {node_dict.get('id')} (storage error): {e}")

        # Restore relationships
        relationships_restored = 0
        for rel_dict in content_dict.get("relationships", []):
            try:
                await self._restore_relationship(rel_dict)
                relationships_restored += 1
            except (KeyError, TypeError, ValueError) as e:
                errors.append(f"Failed to restore relationship (data error): {e}")
            except (OSError, ConnectionError, RuntimeError) as e:
                errors.append(f"Failed to restore relationship (storage error): {e}")

        # Restore culture patterns
        culture_restored = False
        if metadata.includes_culture and content_dict.get("culture_patterns"):
            try:
                await self._restore_culture(content_dict["culture_patterns"])
                culture_restored = True
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                errors.append(f"Failed to restore culture patterns: {e}")
                logger.warning(f"Culture pattern restore failed: {e}")

        # Restore staleness state
        if metadata.includes_staleness and content_dict.get("staleness_state"):
            try:
                await self._restore_staleness(content_dict["staleness_state"])
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                errors.append(f"Failed to restore staleness state: {e}")
                logger.warning(f"Staleness state restore failed: {e}")

        # Restore vectors if requested and present
        if restore_vectors and metadata.includes_vectors and content_dict.get("vector_embeddings"):
            try:
                await self._restore_vectors(content_dict["vector_embeddings"])
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                errors.append(f"Failed to restore vectors: {e}")
                logger.warning(f"Vector restore failed: {e}")

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Restored KM checkpoint {checkpoint_id}: "
            f"{nodes_restored} nodes, {relationships_restored} relationships, "
            f"{len(errors)} errors, {duration_ms}ms"
        )

        return RestoreResult(
            success=len(errors) == 0 or nodes_restored > 0,
            checkpoint_id=checkpoint_id,
            nodes_restored=nodes_restored,
            relationships_restored=relationships_restored,
            culture_restored=culture_restored,
            duration_ms=duration_ms,
            errors=errors,
        )

    async def list_checkpoints(self) -> List[KMCheckpointMetadata]:
        """List all available checkpoints."""
        checkpoints = []

        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("km_ckpt_"):
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    try:
                        metadata_dict = json.loads(metadata_file.read_text())
                        checkpoints.append(KMCheckpointMetadata(**metadata_dict))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to read checkpoint metadata {item.name}: {e}")

        # Sort by creation time (newest first)
        checkpoints.sort(key=lambda x: x.created_at, reverse=True)

        return checkpoints

    async def get_checkpoint_metadata(self, checkpoint_id: str) -> Optional[KMCheckpointMetadata]:
        """Get metadata for a specific checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        metadata_file = checkpoint_path / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            metadata_dict = json.loads(metadata_file.read_text())
            return KMCheckpointMetadata(**metadata_dict)
        except (json.JSONDecodeError, TypeError):
            return None

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_id

        if not checkpoint_path.exists():
            return False

        try:
            shutil.rmtree(checkpoint_path)
            logger.info(f"Deleted KM checkpoint: {checkpoint_id}")
            return True
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    async def compare_checkpoints(
        self,
        checkpoint_id_1: str,
        checkpoint_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two checkpoints.

        Returns statistics about differences between checkpoints.
        """
        meta1 = await self.get_checkpoint_metadata(checkpoint_id_1)
        meta2 = await self.get_checkpoint_metadata(checkpoint_id_2)

        if not meta1 or not meta2:
            return {"error": "One or both checkpoints not found"}

        return {
            "checkpoint_1": checkpoint_id_1,
            "checkpoint_2": checkpoint_id_2,
            "node_count_diff": meta2.node_count - meta1.node_count,
            "relationship_count_diff": meta2.relationship_count - meta1.relationship_count,
            "size_diff_bytes": meta2.size_bytes - meta1.size_bytes,
            "time_diff_seconds": (
                datetime.fromisoformat(meta2.created_at) -
                datetime.fromisoformat(meta1.created_at)
            ).total_seconds(),
        }

    # =========================================================================
    # Private Export Methods
    # =========================================================================

    async def _export_nodes(
        self,
        incremental: bool,
        parent_checkpoint_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Export nodes from KM."""
        nodes = []

        # Get all nodes from the meta store
        if hasattr(self.mound, "_meta_store") and self.mound._meta_store:
            if hasattr(self.mound._meta_store, "query_nodes"):
                raw_nodes = self.mound._meta_store.query_nodes(
                    workspace_id=self.mound.workspace_id,
                    limit=100000,  # High limit for full export
                )
                for node in raw_nodes:
                    node_dict = {
                        "id": node.id,
                        "node_type": node.node_type,
                        "content": node.content,
                        "confidence": node.confidence,
                        "workspace_id": node.workspace_id,
                        "metadata": node.metadata or {},
                        "topics": node.topics or [],
                    }
                    if hasattr(node, "created_at") and node.created_at:
                        node_dict["created_at"] = node.created_at.isoformat() if hasattr(node.created_at, "isoformat") else str(node.created_at)
                    if hasattr(node, "provenance") and node.provenance:
                        node_dict["provenance"] = {
                            "source_type": node.provenance.source_type.value if hasattr(node.provenance.source_type, "value") else str(node.provenance.source_type),
                            "source_id": node.provenance.source_id,
                        }
                    nodes.append(node_dict)

        return nodes

    async def _export_relationships(
        self,
        incremental: bool,
        parent_checkpoint_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Export relationships from KM."""
        relationships = []

        # Get relationships from meta store
        if hasattr(self.mound, "_meta_store") and self.mound._meta_store:
            if hasattr(self.mound._meta_store, "get_all_relationships"):
                raw_rels = self.mound._meta_store.get_all_relationships()
                for rel in raw_rels:
                    relationships.append({
                        "from_node_id": rel.from_node_id,
                        "to_node_id": rel.to_node_id,
                        "relationship_type": rel.relationship_type,
                        "metadata": getattr(rel, "metadata", {}),
                    })

        return relationships

    async def _export_culture(self) -> Dict[str, Any]:
        """Export culture accumulator state."""
        if hasattr(self.mound, "_culture_accumulator") and self.mound._culture_accumulator:
            try:
                return self.mound._culture_accumulator.export_state()
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Culture export skipped due to error: {e}")
        return {}

    async def _export_staleness(self) -> Dict[str, Any]:
        """Export staleness detector state."""
        if hasattr(self.mound, "_staleness_detector") and self.mound._staleness_detector:
            try:
                return self.mound._staleness_detector.export_state()
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Staleness export skipped due to error: {e}")
        return {}

    async def _export_vectors(self) -> Dict[str, List[float]]:
        """Export vector embeddings."""
        vectors = {}

        if hasattr(self.mound, "_semantic_store") and self.mound._semantic_store:
            try:
                vectors = self.mound._semantic_store.export_embeddings()
            except (AttributeError, TypeError, ValueError, OSError) as e:
                logger.debug(f"Vector export skipped due to error: {e}")

        return vectors

    # =========================================================================
    # Private Restore Methods
    # =========================================================================

    async def _clear_mound_data(self) -> None:
        """Clear existing KM data before restore."""
        # This is dangerous - should be done carefully
        logger.warning("Clearing existing KM data for restore")

        if hasattr(self.mound, "_meta_store") and self.mound._meta_store:
            if hasattr(self.mound._meta_store, "clear_workspace"):
                self.mound._meta_store.clear_workspace(self.mound.workspace_id)

    async def _restore_node(self, node_dict: Dict[str, Any]) -> None:
        """Restore a single node."""
        await self.mound._save_node(node_dict)

    async def _restore_relationship(self, rel_dict: Dict[str, Any]) -> None:
        """Restore a single relationship."""
        await self.mound._save_relationship(
            from_id=rel_dict["from_node_id"],
            to_id=rel_dict["to_node_id"],
            rel_type=rel_dict["relationship_type"],
        )

    async def _restore_culture(self, culture_state: Dict[str, Any]) -> None:
        """Restore culture accumulator state."""
        if hasattr(self.mound, "_culture_accumulator") and self.mound._culture_accumulator:
            try:
                self.mound._culture_accumulator.import_state(culture_state)
            except Exception as e:
                logger.warning(f"Failed to restore culture state: {e}")

    async def _restore_staleness(self, staleness_state: Dict[str, Any]) -> None:
        """Restore staleness detector state."""
        if hasattr(self.mound, "_staleness_detector") and self.mound._staleness_detector:
            try:
                self.mound._staleness_detector.import_state(staleness_state)
            except Exception as e:
                logger.warning(f"Failed to restore staleness state: {e}")

    async def _restore_vectors(self, vectors: Dict[str, List[float]]) -> None:
        """Restore vector embeddings."""
        if hasattr(self.mound, "_semantic_store") and self.mound._semantic_store:
            try:
                self.mound._semantic_store.import_embeddings(vectors)
            except Exception as e:
                logger.warning(f"Failed to restore vectors: {e}")

    async def _prune_old_checkpoints(self) -> int:
        """Prune old checkpoints beyond max_checkpoints."""
        checkpoints = await self.list_checkpoints()

        if len(checkpoints) <= self.max_checkpoints:
            return 0

        # Delete oldest checkpoints
        to_delete = checkpoints[self.max_checkpoints:]
        deleted = 0

        for ckpt in to_delete:
            if await self.delete_checkpoint(ckpt.id):
                deleted += 1

        if deleted > 0:
            logger.info(f"Pruned {deleted} old KM checkpoints")

        return deleted


# ============================================================================
# Singleton and Factory
# ============================================================================

_checkpoint_store: Optional[KMCheckpointStore] = None


def get_km_checkpoint_store(
    mound: Optional["KnowledgeMound"] = None,
    checkpoint_dir: Optional[str] = None,
) -> Optional[KMCheckpointStore]:
    """
    Get or create the KM checkpoint store singleton.

    Args:
        mound: KnowledgeMound instance (required for first call)
        checkpoint_dir: Optional checkpoint directory override

    Returns:
        KMCheckpointStore instance or None if mound not provided
    """
    global _checkpoint_store

    if _checkpoint_store is None and mound is not None:
        _checkpoint_store = KMCheckpointStore(mound, checkpoint_dir)

    return _checkpoint_store


def reset_km_checkpoint_store() -> None:
    """Reset the checkpoint store singleton (for testing)."""
    global _checkpoint_store
    _checkpoint_store = None


__all__ = [
    "KMCheckpointStore",
    "KMCheckpointMetadata",
    "KMCheckpointContent",
    "RestoreResult",
    "get_km_checkpoint_store",
    "reset_km_checkpoint_store",
]

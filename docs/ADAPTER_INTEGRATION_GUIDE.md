# Knowledge Mound Adapter Integration Guide

This guide explains how to integrate with the Knowledge Mound adapter system and create new adapters.

## Overview

Knowledge Mound adapters bridge Aragora subsystems to unified knowledge storage. Each adapter:

- Syncs data from a source system to Knowledge Mound nodes
- Provides semantic search over the adapted data
- Emits events for real-time updates
- Supports batch operations for efficient bulk sync

## Available Adapters

| Adapter | Source System | Node Type |
|---------|---------------|-----------|
| `ContinuumAdapter` | ContinuumMemory | memory |
| `ConsensusAdapter` | ConsensusMemory | consensus |
| `CritiqueAdapter` | CritiqueStore | critique |
| `EvidenceAdapter` | EvidenceCollector | evidence |
| `PulseAdapter` | PulseClient | pulse |
| `InsightsAdapter` | InsightExtractor | insight |
| `EloAdapter` | EloSystem | ranking |
| `BeliefAdapter` | BeliefNetwork | belief |
| `CostAdapter` | CostTracker | cost |
| `ReceiptAdapter` | ReceiptManager | receipt |
| `ControlPlaneAdapter` | ControlPlane | control |
| `RLMAdapter` | RLMHandler | rlm |
| `CultureAdapter` | CultureProfile | culture |
| `RankingAdapter` | RankingSystem | ranking |
| `CalibrationFusionAdapter` | CalibrationFusionEngine | calibration |

## Adapter Factory

The adapter factory auto-creates adapters from Arena subsystems:

```python
from aragora.knowledge.mound.adapters import AdapterFactory

# Create factory with Knowledge Mound instance
factory = AdapterFactory(knowledge_mound)

# Auto-detect and create adapters from Arena
adapters = factory.create_from_arena(arena)

# Or create specific adapters
continuum_adapter = factory.create_continuum_adapter(arena.continuum_memory)
elo_adapter = factory.create_elo_adapter(arena.elo_system)
```

### Factory Methods

```python
class AdapterFactory:
    def create_from_arena(self, arena: Arena) -> list[KnowledgeMoundAdapter]:
        """Create all applicable adapters from Arena subsystems."""

    def create_continuum_adapter(self, memory: ContinuumMemory) -> ContinuumAdapter:
        """Create adapter for ContinuumMemory."""

    def create_consensus_adapter(self, memory: ConsensusMemory) -> ConsensusAdapter:
        """Create adapter for ConsensusMemory."""

    def create_elo_adapter(self, elo: EloSystem) -> EloAdapter:
        """Create adapter for ELO ranking system."""

    def create_calibration_adapter(
        self,
        engine: CalibrationFusionEngine
    ) -> CalibrationFusionAdapter:
        """Create adapter for calibration fusion."""
```

## Base Adapter Interface

All adapters implement `KnowledgeMoundAdapter`:

```python
from aragora.knowledge.mound.adapters._base import (
    KnowledgeMoundAdapter,
    EventCallback,
    SyncResult,
)

class MyAdapter(KnowledgeMoundAdapter):
    """Adapter for MySubsystem."""

    ADAPTER_NAME = "my_adapter"
    NODE_TYPE = "my_node_type"

    def __init__(
        self,
        subsystem: MySubsystem,
        mound: Optional[KnowledgeMound] = None,
    ):
        super().__init__(mound)
        self.subsystem = subsystem

    async def sync_to_mound(
        self,
        workspace_id: str = "default",
        incremental: bool = True,
        since: Optional[datetime] = None,
    ) -> SyncResult:
        """Sync subsystem data to Knowledge Mound."""
        ...

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[KnowledgeItem]:
        """Search adapter's knowledge nodes."""
        ...

    def to_knowledge_item(self, source_item: Any) -> KnowledgeItem:
        """Convert source item to KnowledgeItem."""
        ...
```

### Required Methods

| Method | Purpose |
|--------|---------|
| `sync_to_mound()` | Sync source data to Knowledge Mound |
| `search()` | Semantic search over adapted data |
| `to_knowledge_item()` | Convert source format to KnowledgeItem |

### Optional Methods

| Method | Purpose |
|--------|---------|
| `batch_sync()` | Batch sync for large datasets |
| `get_by_id()` | Retrieve single item by ID |
| `delete()` | Remove synced item |
| `on_source_update()` | Handle source system updates |

## Creating a New Adapter

### 1. Define the Adapter

```python
# aragora/knowledge/mound/adapters/my_adapter.py

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from aragora.knowledge.mound.adapters._base import (
    KnowledgeMoundAdapter,
    SyncResult,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound
    from aragora.knowledge.mound.types import KnowledgeItem
    from aragora.mysubsystem import MySubsystem


class MyAdapter(KnowledgeMoundAdapter):
    """Adapter for MySubsystem data."""

    ADAPTER_NAME = "my_adapter"
    NODE_TYPE = "my_type"

    def __init__(
        self,
        subsystem: "MySubsystem",
        mound: Optional["KnowledgeMound"] = None,
    ):
        super().__init__(mound)
        self.subsystem = subsystem
        self._last_sync: Optional[datetime] = None

    async def sync_to_mound(
        self,
        workspace_id: str = "default",
        incremental: bool = True,
        since: Optional[datetime] = None,
    ) -> SyncResult:
        """Sync subsystem data to Knowledge Mound."""
        if not self._mound:
            return SyncResult(synced=0, errors=["No mound configured"])

        # Determine sync window
        sync_since = since
        if incremental and not sync_since:
            sync_since = self._last_sync

        # Get items from source
        items = await self.subsystem.get_items(since=sync_since)

        # Convert and store
        synced = 0
        errors = []
        for item in items:
            try:
                knowledge_item = self.to_knowledge_item(item)
                await self._mound.add_node(
                    workspace_id=workspace_id,
                    **knowledge_item,
                )
                synced += 1
            except Exception as e:
                errors.append(f"Failed to sync {item.id}: {e}")

        self._last_sync = datetime.now()
        return SyncResult(synced=synced, errors=errors)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list["KnowledgeItem"]:
        """Search adapted knowledge nodes."""
        if not self._mound:
            return []

        return await self._mound.query(
            query=query,
            limit=limit,
            node_type=self.NODE_TYPE,
            filters=filters,
        )

    def to_knowledge_item(self, source_item: Any) -> dict:
        """Convert source item to KnowledgeItem dict."""
        return {
            "id": f"{self.ADAPTER_NAME}:{source_item.id}",
            "content": source_item.content,
            "node_type": self.NODE_TYPE,
            "source": self.ADAPTER_NAME,
            "metadata": {
                "original_id": source_item.id,
                "created_at": source_item.created_at.isoformat(),
            },
        }
```

### 2. Register in `__init__.py`

```python
# aragora/knowledge/mound/adapters/__init__.py

from .my_adapter import MyAdapter

__all__ = [
    # ... existing adapters ...
    "MyAdapter",
]
```

### 3. Add Factory Support

```python
# aragora/knowledge/mound/adapters/factory.py

def create_my_adapter(
    self,
    subsystem: "MySubsystem",
) -> MyAdapter:
    """Create adapter for MySubsystem."""
    return MyAdapter(subsystem, self._mound)
```

### 4. Write Tests

```python
# tests/knowledge/mound/adapters/test_my_adapter.py

import pytest
from aragora.knowledge.mound.adapters import MyAdapter


class TestMyAdapter:
    @pytest.fixture
    def adapter(self, mock_mound, mock_subsystem):
        return MyAdapter(mock_subsystem, mock_mound)

    async def test_sync_to_mound(self, adapter):
        result = await adapter.sync_to_mound("workspace_1")
        assert result.synced > 0
        assert len(result.errors) == 0

    async def test_search(self, adapter):
        # First sync some data
        await adapter.sync_to_mound()

        # Then search
        results = await adapter.search("test query", limit=5)
        assert len(results) <= 5

    def test_to_knowledge_item(self, adapter):
        source = MockItem(id="123", content="test content")
        item = adapter.to_knowledge_item(source)

        assert item["id"] == "my_adapter:123"
        assert item["node_type"] == "my_type"
        assert item["content"] == "test content"
```

## Batch Operations

For adapters handling large datasets, implement batch sync:

```python
async def batch_sync(
    self,
    workspace_id: str = "default",
    batch_size: int = 100,
    max_batches: Optional[int] = None,
) -> SyncResult:
    """Sync data in batches for memory efficiency."""
    total_synced = 0
    all_errors = []
    batch_count = 0

    async for batch in self.subsystem.iter_batches(size=batch_size):
        if max_batches and batch_count >= max_batches:
            break

        for item in batch:
            try:
                knowledge_item = self.to_knowledge_item(item)
                await self._mound.add_node(
                    workspace_id=workspace_id,
                    **knowledge_item,
                )
                total_synced += 1
            except Exception as e:
                all_errors.append(str(e))

        batch_count += 1
        logger.info(f"Synced batch {batch_count}: {len(batch)} items")

    return SyncResult(synced=total_synced, errors=all_errors)
```

## Event Handling

Adapters can subscribe to source system events:

```python
def __init__(self, subsystem, mound):
    super().__init__(mound)
    self.subsystem = subsystem

    # Subscribe to source updates
    if hasattr(subsystem, "on_update"):
        subsystem.on_update(self._handle_update)

async def _handle_update(self, event: dict) -> None:
    """Handle source system update event."""
    if event["type"] == "created":
        item = event["item"]
        await self._mound.add_node(
            workspace_id="default",
            **self.to_knowledge_item(item),
        )
    elif event["type"] == "deleted":
        await self._mound.delete_node(
            node_id=f"{self.ADAPTER_NAME}:{event['item_id']}",
        )
```

## Best Practices

1. **Incremental sync**: Always support `since` parameter for efficient updates
2. **Idempotent operations**: Use stable IDs to allow re-sync without duplicates
3. **Error handling**: Collect errors but continue processing
4. **Metadata preservation**: Store original IDs and timestamps in metadata
5. **Batch large datasets**: Use batch operations for memory efficiency
6. **Test thoroughly**: Cover sync, search, and edge cases

## Related Documentation

- [Knowledge Mound Overview](../aragora/knowledge/mound/README.md)
- [Handler Architecture](../aragora/server/handlers/HANDLER_ARCHITECTURE.md)
- [Phase A3 Features](./STATUS.md)

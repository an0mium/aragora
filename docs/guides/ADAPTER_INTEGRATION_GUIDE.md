# Knowledge Mound Adapter Integration Guide

This guide explains how to integrate with the Knowledge Mound adapter system and create new adapters.

## Overview

Knowledge Mound adapters bridge Aragora subsystems to unified knowledge storage. Each adapter:

- Syncs data from a source system to Knowledge Mound nodes
- Provides semantic search over the adapted data
- Emits events for real-time updates
- Supports batch operations for efficient bulk sync

## Available Adapters

| Adapter | Source System | Node Type | Mixins |
|---------|---------------|-----------|--------|
| `ContinuumAdapter` | ContinuumMemory | memory | Semantic |
| `ConsensusAdapter` | ConsensusMemory | consensus | Semantic |
| `CritiqueAdapter` | CritiqueStore | critique | Semantic |
| `EvidenceAdapter` | EvidenceCollector | evidence | Semantic |
| `PulseAdapter` | PulseClient | pulse | Semantic |
| `InsightsAdapter` | InsightExtractor | insight | Semantic |
| `EloAdapter` | EloSystem | ranking | - |
| `BeliefAdapter` | BeliefNetwork | belief | Semantic, ReverseFlow |
| `CostAdapter` | CostTracker | cost | - |
| `ReceiptAdapter` | ReceiptManager | receipt | Semantic |
| `ControlPlaneAdapter` | ControlPlane | control | - |
| `RLMAdapter` | RLMHandler | rlm | Semantic |
| `CultureAdapter` | CultureProfile | culture | Semantic |
| `RankingAdapter` | RankingSystem | ranking | - |
| `CalibrationFusionAdapter` | CalibrationFusionEngine | calibration | Fusion |
| `ComputerUseAdapter` | ComputerUseAgent | computer_use | Semantic |
| `FabricAdapter` | FabricClient | fabric | Semantic |
| `GatewayAdapter` | GatewayRouter | gateway | - |
| `PerformanceAdapter` | PerformanceMonitor | performance | - |
| `ProvenanceAdapter` | ProvenanceTracker | provenance | Semantic, ReverseFlow |
| `WorkspaceAdapter` | WorkspaceManager | workspace | Semantic |

**Total: 21 adapters** supporting various subsystems with optional mixin composition.

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

## Mixin Composition Pattern

Adapters can be composed with mixins for additional capabilities:

### Available Mixins

| Mixin | Purpose | Location |
|-------|---------|----------|
| `SemanticSearchMixin` | Adds vector-based semantic search | `_semantic_mixin.py` |
| `ReverseFlowMixin` | Enables writing back to source systems | `_reverse_flow_base.py` |
| `FusionMixin` | Combines multiple adapter results | `_fusion_mixin.py` |

### SemanticSearchMixin

Adds embedding-based semantic search to any adapter:

```python
from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin

class MyAdapter(KnowledgeMoundAdapter, SemanticSearchMixin):
    """Adapter with semantic search capability."""

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[KnowledgeItem]:
        # SemanticSearchMixin provides semantic_search()
        return await self.semantic_search(
            query=query,
            limit=limit,
            node_type=self.NODE_TYPE,
            filters=filters,
        )
```

### ReverseFlowMixin

Enables adapters to write knowledge back to source systems:

```python
from aragora.knowledge.mound.adapters._reverse_flow_base import ReverseFlowMixin

class BeliefAdapter(KnowledgeMoundAdapter, SemanticSearchMixin, ReverseFlowMixin):
    """Adapter that can update BeliefNetwork from Knowledge Mound."""

    async def write_back(
        self,
        node_id: str,
        updates: dict,
    ) -> bool:
        """Write Knowledge Mound updates back to BeliefNetwork."""
        knowledge_item = await self._mound.get_node(node_id)
        if not knowledge_item:
            return False

        # Update source system
        belief_id = knowledge_item.metadata.get("original_id")
        await self.subsystem.update_belief(belief_id, updates)
        return True
```

### FusionMixin

Combines results from multiple adapters for cross-system queries:

```python
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin

class CalibrationFusionAdapter(KnowledgeMoundAdapter, FusionMixin):
    """Fuses calibration data from multiple sources."""

    def __init__(self, engine, mound, adapters: list[KnowledgeMoundAdapter]):
        super().__init__(mound)
        self.engine = engine
        self._adapters = adapters

    async def fused_search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[KnowledgeItem]:
        """Search across all fused adapters."""
        all_results = []
        for adapter in self._adapters:
            results = await adapter.search(query, limit=limit)
            all_results.extend(results)

        # Apply fusion ranking
        return self._rank_fused_results(all_results, limit)
```

### Composition Example

Building an adapter with multiple mixins:

```python
class ProvenanceAdapter(
    KnowledgeMoundAdapter,
    SemanticSearchMixin,
    ReverseFlowMixin,
):
    """Provenance tracking with semantic search and bidirectional sync."""

    ADAPTER_NAME = "provenance"
    NODE_TYPE = "provenance"

    def __init__(self, tracker, mound):
        # Initialize all base classes
        KnowledgeMoundAdapter.__init__(self, mound)
        SemanticSearchMixin.__init__(self)
        ReverseFlowMixin.__init__(self)

        self.tracker = tracker

    # Methods from all mixins are available:
    # - sync_to_mound() from KnowledgeMoundAdapter
    # - semantic_search() from SemanticSearchMixin
    # - write_back() from ReverseFlowMixin
```

## Resilience Integration

Adapters can use Aragora's resilience patterns for fault tolerance:

### Retry with Backoff

```python
from aragora.resilience import with_retry, RetryConfig

class MyAdapter(KnowledgeMoundAdapter):
    async def sync_to_mound(self, workspace_id: str = "default", **kwargs):
        @with_retry(RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential=True,
        ))
        async def _do_sync():
            items = await self.subsystem.get_items()
            for item in items:
                await self._mound.add_node(
                    workspace_id=workspace_id,
                    **self.to_knowledge_item(item),
                )
            return len(items)

        synced = await _do_sync()
        return SyncResult(synced=synced, errors=[])
```

### Circuit Breaker

```python
from aragora.resilience import with_circuit_breaker, CircuitBreakerConfig

class MyAdapter(KnowledgeMoundAdapter):
    def __init__(self, subsystem, mound):
        super().__init__(mound)
        self.subsystem = subsystem
        self._circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_requests=2,
        )

    @with_circuit_breaker("my_adapter_sync")
    async def sync_to_mound(self, workspace_id: str = "default", **kwargs):
        # Protected by circuit breaker
        ...
```

### Timeout Protection

```python
from aragora.resilience import with_timeout, TimeoutConfig

class MyAdapter(KnowledgeMoundAdapter):
    @with_timeout(TimeoutConfig(timeout_seconds=30.0))
    async def search(self, query: str, limit: int = 10, **kwargs):
        # Will timeout after 30 seconds
        return await self._mound.query(query=query, limit=limit)
```

### Combined Resilience

Use the resilience context manager for combined patterns:

```python
from aragora.knowledge.mound.resilience import resilient_operation

class MyAdapter(KnowledgeMoundAdapter):
    async def sync_to_mound(self, workspace_id: str = "default", **kwargs):
        async with resilient_operation(
            operation_name="my_adapter_sync",
            retry_config=RetryConfig(max_retries=3),
            timeout_seconds=60.0,
            circuit_breaker_name="my_adapter",
        ):
            # All resilience patterns applied
            items = await self.subsystem.get_items()
            ...
```

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

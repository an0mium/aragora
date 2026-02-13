# Knowledge Mound Adapter Guide

This guide explains how to create adapters that integrate external systems with Aragora's Knowledge Mound.

## Overview

The **Knowledge Mound** is Aragora's unified knowledge storage system, implementing a "termite mound" architecture where all agents contribute to and query from a shared knowledge superstructure. Adapters allow you to:

- Sync knowledge from external systems (databases, APIs, file systems)
- Maintain provenance tracking for compliance
- Enable cross-system queries
- Implement custom validation and transformation logic

## Quick Start

### 1. Create a Basic Adapter

```python
from aragora.knowledge.mound.adapters.base import KnowledgeAdapter, AdapterConfig
from aragora.knowledge.mound.types import KnowledgeItem, IngestionResult, QueryResult

class SlackAdapter(KnowledgeAdapter):
    """Adapter for syncing knowledge from Slack conversations."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.client = None

    @property
    def adapter_type(self) -> str:
        return "slack"

    @property
    def supported_sources(self) -> list[str]:
        return ["slack_message", "slack_thread", "slack_file"]

    async def initialize(self) -> None:
        """Initialize the Slack client."""
        from slack_sdk.web.async_client import AsyncWebClient

        token = self.config.credentials.get("token")
        if not token:
            raise ValueError("Slack token required in credentials")

        self.client = AsyncWebClient(token=token)
        await self._verify_connection()

    async def _verify_connection(self) -> None:
        """Verify Slack connection works."""
        response = await self.client.auth_test()
        if not response["ok"]:
            raise ConnectionError(f"Slack auth failed: {response.get('error')}")

    async def sync(
        self,
        workspace_id: str,
        since: datetime | None = None,
    ) -> IngestionResult:
        """Sync messages from Slack channels."""
        items = []

        # Get channels
        channels = await self._get_channels()

        for channel in channels:
            messages = await self._fetch_messages(channel["id"], since)
            for msg in messages:
                item = self._message_to_knowledge_item(msg, channel)
                items.append(item)

        # Store in Knowledge Mound
        return await self.mound.batch_store(items, workspace_id=workspace_id)

    async def query(
        self,
        query: str,
        workspace_id: str,
        limit: int = 10,
    ) -> QueryResult:
        """Query knowledge from Slack."""
        return await self.mound.query(
            query=query,
            workspace_id=workspace_id,
            source_filter=["slack_message", "slack_thread"],
            limit=limit,
        )

    def _message_to_knowledge_item(self, msg: dict, channel: dict) -> KnowledgeItem:
        """Convert Slack message to KnowledgeItem."""
        return KnowledgeItem(
            content=msg["text"],
            source_type="slack_message",
            source_id=f"slack:{channel['id']}:{msg['ts']}",
            metadata={
                "channel": channel["name"],
                "user": msg.get("user"),
                "timestamp": msg["ts"],
                "reactions": msg.get("reactions", []),
            },
            created_at=datetime.fromtimestamp(float(msg["ts"])),
        )
```

### 2. Register Your Adapter

```python
from aragora.knowledge.mound.adapters import register_adapter

# Register during application startup
register_adapter("slack", SlackAdapter)
```

### 3. Use Your Adapter

```python
from aragora.knowledge.mound.adapters import get_adapter

# Get adapter instance
adapter = get_adapter("slack", config=AdapterConfig(
    credentials={"token": "xoxb-..."},
    sync_interval_minutes=30,
))

# Initialize
await adapter.initialize()

# Sync knowledge
result = await adapter.sync(workspace_id="team-engineering")
print(f"Synced {result.items_stored} items")

# Query
results = await adapter.query("deployment process", workspace_id="team-engineering")
for item in results.items:
    print(f"- {item.content[:100]}...")
```

## Adapter Base Class

All adapters extend `KnowledgeAdapter`:

```python
class KnowledgeAdapter(ABC):
    """Base class for Knowledge Mound adapters."""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.mound = None  # Set during initialization

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """Unique identifier for this adapter type."""
        pass

    @property
    @abstractmethod
    def supported_sources(self) -> list[str]:
        """List of source types this adapter handles."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize adapter (connect to external system)."""
        pass

    @abstractmethod
    async def sync(
        self,
        workspace_id: str,
        since: datetime | None = None,
    ) -> IngestionResult:
        """Sync knowledge from external system."""
        pass

    async def query(
        self,
        query: str,
        workspace_id: str,
        limit: int = 10,
    ) -> QueryResult:
        """Query knowledge (default: delegate to mound)."""
        return await self.mound.query(
            query=query,
            workspace_id=workspace_id,
            source_filter=self.supported_sources,
            limit=limit,
        )

    async def validate(self, item: KnowledgeItem) -> ValidationResult:
        """Validate a knowledge item (override for custom validation)."""
        return ValidationResult(valid=True)

    async def transform(self, raw_data: Any) -> KnowledgeItem:
        """Transform raw data to KnowledgeItem (override for custom logic)."""
        raise NotImplementedError

    async def close(self) -> None:
        """Cleanup resources."""
        pass
```

## Configuration

Adapters receive configuration via `AdapterConfig`:

```python
@dataclass
class AdapterConfig:
    # Connection settings
    credentials: dict[str, str]  # API keys, tokens, passwords
    endpoint: str | None = None  # Custom endpoint URL

    # Sync settings
    sync_interval_minutes: int = 60
    batch_size: int = 100
    max_items_per_sync: int = 10000

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Filter settings
    include_patterns: list[str] | None = None  # Glob patterns to include
    exclude_patterns: list[str] | None = None  # Glob patterns to exclude

    # Workspace mapping
    workspace_id: str | None = None  # Default workspace

    # Feature flags
    enable_embeddings: bool = True
    enable_provenance: bool = True
    enable_staleness_check: bool = True
```

## Knowledge Items

Adapters produce `KnowledgeItem` objects:

```python
@dataclass
class KnowledgeItem:
    # Required
    content: str              # The knowledge content
    source_type: str          # Type identifier (e.g., "slack_message")
    source_id: str            # Unique source identifier

    # Optional metadata
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Provenance
    author_id: str | None = None
    author_name: str | None = None

    # Embedding (computed if not provided)
    embedding: list[float] | None = None

    # Relationships
    parent_id: str | None = None
    related_ids: list[str] = field(default_factory=list)

    # Confidence
    confidence: float = 1.0
    validated: bool = False
```

## Built-in Adapters

Aragora includes 25+ built-in adapters:

| Adapter | Source Types | Description |
|---------|-------------|-------------|
| `belief` | debate beliefs | Agent beliefs from debates |
| `consensus` | consensus decisions | Consensus outcomes |
| `continuum` | memory entries | Multi-tier memory items |
| `critique` | debate critiques | Agent critiques |
| `culture` | cultural patterns | Organizational patterns |
| `elo` | agent rankings | Agent ELO scores |
| `evidence` | collected evidence | Evidence from debates |
| `insights` | debate insights | Generated insights |
| `performance` | agent metrics | Performance data |
| `provenance` | claim chains | Provenance tracking |
| `pulse` | trending topics | Pulse system topics |
| `workspace` | workspace data | Workspace metadata |

## Implementing Common Patterns

### Database Adapter

```python
class PostgresAdapter(KnowledgeAdapter):
    """Adapter for PostgreSQL databases."""

    @property
    def adapter_type(self) -> str:
        return "postgres"

    @property
    def supported_sources(self) -> list[str]:
        return ["postgres_row", "postgres_document"]

    async def initialize(self) -> None:
        import asyncpg

        self.pool = await asyncpg.create_pool(
            self.config.credentials["dsn"],
            min_size=2,
            max_size=10,
        )

    async def sync(
        self,
        workspace_id: str,
        since: datetime | None = None,
    ) -> IngestionResult:
        query = self.config.credentials.get("query")
        if not query:
            raise ValueError("SQL query required in credentials")

        async with self.pool.acquire() as conn:
            if since:
                rows = await conn.fetch(query, since)
            else:
                rows = await conn.fetch(query)

            items = [self._row_to_item(row) for row in rows]
            return await self.mound.batch_store(items, workspace_id=workspace_id)

    def _row_to_item(self, row: asyncpg.Record) -> KnowledgeItem:
        return KnowledgeItem(
            content=str(row.get("content", row)),
            source_type="postgres_row",
            source_id=f"pg:{row.get('id', hash(tuple(row.values())))}",
            metadata=dict(row),
        )
```

### API Adapter

```python
class NotionAdapter(KnowledgeAdapter):
    """Adapter for Notion pages and databases."""

    @property
    def adapter_type(self) -> str:
        return "notion"

    @property
    def supported_sources(self) -> list[str]:
        return ["notion_page", "notion_database"]

    async def initialize(self) -> None:
        import httpx

        self.client = httpx.AsyncClient(
            base_url="https://api.notion.com/v1",
            headers={
                "Authorization": f"Bearer {self.config.credentials['token']}",
                "Notion-Version": "2022-06-28",
            },
        )

    async def sync(
        self,
        workspace_id: str,
        since: datetime | None = None,
    ) -> IngestionResult:
        items = []

        # Sync pages
        pages = await self._list_pages()
        for page in pages:
            content = await self._get_page_content(page["id"])
            items.append(KnowledgeItem(
                content=content,
                source_type="notion_page",
                source_id=f"notion:{page['id']}",
                metadata={
                    "title": page.get("properties", {}).get("title", "Untitled"),
                    "url": page.get("url"),
                },
                created_at=datetime.fromisoformat(page["created_time"].rstrip("Z")),
            ))

        return await self.mound.batch_store(items, workspace_id=workspace_id)
```

### File System Adapter

```python
class FileSystemAdapter(KnowledgeAdapter):
    """Adapter for local file systems."""

    @property
    def adapter_type(self) -> str:
        return "filesystem"

    @property
    def supported_sources(self) -> list[str]:
        return ["file_text", "file_markdown", "file_pdf"]

    async def initialize(self) -> None:
        self.root_path = Path(self.config.credentials["path"])
        if not self.root_path.exists():
            raise ValueError(f"Path does not exist: {self.root_path}")

    async def sync(
        self,
        workspace_id: str,
        since: datetime | None = None,
    ) -> IngestionResult:
        items = []

        for pattern in self.config.include_patterns or ["**/*.md", "**/*.txt"]:
            for file_path in self.root_path.glob(pattern):
                if self._should_include(file_path, since):
                    item = await self._file_to_item(file_path)
                    items.append(item)

        return await self.mound.batch_store(items, workspace_id=workspace_id)

    async def _file_to_item(self, path: Path) -> KnowledgeItem:
        content = path.read_text()
        stat = path.stat()

        return KnowledgeItem(
            content=content,
            source_type=f"file_{path.suffix.lstrip('.')}",
            source_id=f"file:{path.absolute()}",
            metadata={
                "filename": path.name,
                "path": str(path.relative_to(self.root_path)),
                "size_bytes": stat.st_size,
            },
            created_at=datetime.fromtimestamp(stat.st_ctime),
            updated_at=datetime.fromtimestamp(stat.st_mtime),
        )
```

## Error Handling

Handle errors gracefully:

```python
class RobustAdapter(KnowledgeAdapter):
    async def sync(
        self,
        workspace_id: str,
        since: datetime | None = None,
    ) -> IngestionResult:
        items = []
        errors = []

        async for raw_item in self._fetch_items(since):
            try:
                item = await self.transform(raw_item)

                # Validate
                validation = await self.validate(item)
                if not validation.valid:
                    errors.append(f"Validation failed: {validation.error}")
                    continue

                items.append(item)

            except Exception as e:
                logger.warning(f"Failed to process item: {e}")
                errors.append(str(e))

        result = await self.mound.batch_store(items, workspace_id=workspace_id)
        result.errors.extend(errors)
        return result
```

## Testing Your Adapter

### Unit Tests

```python
import pytest
from datetime import datetime

@pytest.fixture
def adapter():
    config = AdapterConfig(
        credentials={"token": "test-token"},
    )
    return SlackAdapter(config)

@pytest.mark.asyncio
async def test_message_to_knowledge_item(adapter):
    msg = {
        "text": "Deploy to production at 5pm",
        "ts": "1612345678.000100",
        "user": "U123",
    }
    channel = {"id": "C456", "name": "engineering"}

    item = adapter._message_to_knowledge_item(msg, channel)

    assert item.content == "Deploy to production at 5pm"
    assert item.source_type == "slack_message"
    assert item.source_id == "slack:C456:1612345678.000100"
    assert item.metadata["channel"] == "engineering"

@pytest.mark.asyncio
async def test_sync_empty(adapter, mock_mound):
    adapter.mound = mock_mound
    adapter.client = MockSlackClient(messages=[])

    result = await adapter.sync(workspace_id="test")

    assert result.items_stored == 0
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_sync_cycle():
    # Setup
    config = AdapterConfig(
        credentials={"token": os.environ["SLACK_TEST_TOKEN"]},
    )
    adapter = SlackAdapter(config)
    mound = KnowledgeMound(workspace_id="test")

    await adapter.initialize()
    adapter.mound = mound

    # Sync
    result = await adapter.sync(workspace_id="test")
    assert result.items_stored > 0

    # Query
    query_result = await adapter.query("deployment", workspace_id="test")
    assert len(query_result.items) > 0
```

## Registration and Discovery

### Manual Registration

```python
from aragora.knowledge.mound.adapters import AdapterRegistry

registry = AdapterRegistry()
registry.register("slack", SlackAdapter)
registry.register("notion", NotionAdapter)
```

### Auto-Discovery

Place adapters in `aragora/knowledge/mound/adapters/custom/`:

```
aragora/knowledge/mound/adapters/custom/
    __init__.py
    slack_adapter.py
    notion_adapter.py
```

In each file, export the adapter class:

```python
# slack_adapter.py
from aragora.knowledge.mound.adapters.base import KnowledgeAdapter

class SlackAdapter(KnowledgeAdapter):
    ...

__all__ = ["SlackAdapter"]
ADAPTER_TYPE = "slack"
```

## Scheduling Syncs

Use the built-in scheduler:

```python
from aragora.knowledge.mound.adapters import AdapterScheduler

scheduler = AdapterScheduler()

# Register adapters with sync intervals
scheduler.register("slack", SlackAdapter, config=AdapterConfig(
    credentials={"token": "..."},
    sync_interval_minutes=30,
))

scheduler.register("notion", NotionAdapter, config=AdapterConfig(
    credentials={"token": "..."},
    sync_interval_minutes=60,
))

# Start background syncing
await scheduler.start()
```

## Best Practices

### 1. Implement Incremental Sync

```python
async def sync(self, workspace_id: str, since: datetime | None = None):
    if since:
        # Incremental sync - only fetch new items
        items = await self._fetch_since(since)
    else:
        # Full sync
        items = await self._fetch_all()
    # ...
```

### 2. Handle Rate Limits

```python
async def _fetch_with_backoff(self, url: str) -> dict:
    for attempt in range(self.config.max_retries):
        response = await self.client.get(url)
        if response.status_code == 429:
            delay = self.config.retry_delay_seconds * (2 ** attempt)
            await asyncio.sleep(delay)
            continue
        return response.json()
    raise RateLimitError("Max retries exceeded")
```

### 3. Track Provenance

```python
item = KnowledgeItem(
    content=content,
    source_type="api_response",
    source_id=f"api:{item_id}",
    metadata={
        "api_version": "v2",
        "endpoint": "/data/items",
        "fetched_at": datetime.utcnow().isoformat(),
    },
)
```

### 4. Log Metrics

```python
import logging
from aragora.observability import metrics

logger = logging.getLogger(__name__)

async def sync(self, workspace_id: str, since: datetime | None = None):
    start = time.time()

    try:
        result = await self._do_sync(workspace_id, since)
        metrics.counter("adapter_sync_success", tags={"adapter": self.adapter_type})
        return result
    except Exception as e:
        metrics.counter("adapter_sync_failure", tags={"adapter": self.adapter_type})
        raise
    finally:
        duration = time.time() - start
        metrics.histogram("adapter_sync_duration", duration, tags={"adapter": self.adapter_type})
```

## See Also

- [PLUGIN_DEVELOPMENT.md](../integrations/PLUGIN_DEVELOPMENT.md) - Building Aragora plugins
- [aragora/knowledge/mound/](../../aragora/knowledge/mound/) - Knowledge Mound source
- [tests/knowledge/mound/](../../tests/knowledge/mound/) - Adapter test examples

"""Fixtures for knowledge module tests."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockMetaStore:
    """Mock metadata store for testing."""

    def __init__(self):
        self._nodes: Dict[str, Any] = {}
        self._relationships: List[Dict[str, Any]] = []
        self._content_hashes: Dict[str, str] = {}  # hash -> node_id

    def save_node(self, node: Any) -> None:
        """Save a node."""
        self._nodes[node.id] = node
        # Track content hash
        if hasattr(node, 'content'):
            import hashlib
            content_hash = hashlib.sha256(node.content.encode()).hexdigest()[:32]
            self._content_hashes[content_hash] = node.id

    def get_node(self, node_id: str) -> Optional[Any]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            return True
        return False

    def find_by_content_hash(self, content_hash: str, workspace_id: str) -> Optional[Any]:
        """Find node by content hash."""
        node_id = self._content_hashes.get(content_hash)
        if node_id:
            return self._nodes.get(node_id)
        return None

    def save_relationship(self, rel: Any) -> None:
        """Save a relationship."""
        self._relationships.append({
            "from_node_id": rel.from_node_id,
            "to_node_id": rel.to_node_id,
            "relationship_type": rel.relationship_type,
        })

    def get_relationships(self, node_id: str) -> List[Any]:
        """Get relationships for a node."""
        return [
            r for r in self._relationships
            if r["from_node_id"] == node_id or r["to_node_id"] == node_id
        ]

    def query_nodes(self, workspace_id: str, limit: int = 100) -> List[Any]:
        """Query nodes."""
        return list(self._nodes.values())[:limit]

    def get_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "total_nodes": len(self._nodes),
            "by_type": {},
            "by_tier": {},
            "average_confidence": 0.5,
        }

    def connection(self):
        """Return a mock connection context manager."""
        return MockConnection()


class MockConnection:
    """Mock database connection."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query: str, params: tuple = ()):
        """Execute a query."""
        return MockCursor()

    def commit(self):
        """Commit the transaction."""
        pass


class MockCursor:
    """Mock database cursor."""

    rowcount = 0

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class MockSemanticStore:
    """Mock semantic store for testing."""

    def __init__(self, db_path: str = "", default_tenant_id: str = "default"):
        self._items: Dict[str, Any] = {}
        self.default_tenant_id = default_tenant_id

    def initialize(self):
        """Initialize the store."""
        pass

    async def index_item(
        self,
        source_type: Any,
        source_id: str,
        content: str,
        tenant_id: str,
        domain: str = "general",
        importance: float = 0.5,
    ) -> None:
        """Index an item."""
        self._items[source_id] = {
            "source_type": source_type,
            "content": content,
            "tenant_id": tenant_id,
            "domain": domain,
            "importance": importance,
        }

    async def search_similar(
        self,
        query: str,
        tenant_id: str,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Any]:
        """Search for similar items."""
        results = []
        query_lower = query.lower()
        for source_id, item in self._items.items():
            if tenant_id == item["tenant_id"]:
                # Simple keyword matching for test
                if query_lower in item["content"].lower():
                    mock_result = MagicMock()
                    mock_result.source_id = source_id
                    mock_result.similarity = 0.8
                    results.append(mock_result)
        return results[:limit]


class MockRedisCache:
    """Mock Redis cache for testing."""

    def __init__(self):
        self._nodes: Dict[str, Any] = {}
        self._queries: Dict[str, Any] = {}
        self._culture: Dict[str, Any] = {}

    async def connect(self) -> None:
        """Connect to cache."""
        pass

    async def close(self) -> None:
        """Close connection."""
        pass

    async def get_node(self, node_id: str) -> Optional[Any]:
        """Get cached node."""
        return self._nodes.get(node_id)

    async def set_node(self, node_id: str, node: Any) -> None:
        """Cache a node."""
        self._nodes[node_id] = node

    async def invalidate_node(self, node_id: str) -> None:
        """Invalidate cached node."""
        self._nodes.pop(node_id, None)

    async def get_query(self, cache_key: str) -> Optional[Any]:
        """Get cached query result."""
        return self._queries.get(cache_key)

    async def set_query(self, cache_key: str, result: Any) -> None:
        """Cache a query result."""
        self._queries[cache_key] = result

    async def invalidate_queries(self, workspace_id: str) -> None:
        """Invalidate queries for workspace."""
        self._queries = {
            k: v for k, v in self._queries.items()
            if not k.startswith(f"{workspace_id}:")
        }

    async def get_culture(self, workspace_id: str) -> Optional[Any]:
        """Get cached culture profile."""
        return self._culture.get(workspace_id)

    async def set_culture(self, workspace_id: str, profile: Any) -> None:
        """Cache culture profile."""
        self._culture[workspace_id] = profile


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_mound.db"


@pytest.fixture
def mock_meta_store():
    """Create a mock metadata store."""
    return MockMetaStore()


@pytest.fixture
def mock_semantic_store():
    """Create a mock semantic store."""
    return MockSemanticStore()


@pytest.fixture
def mock_redis_cache():
    """Create a mock Redis cache."""
    return MockRedisCache()


@pytest.fixture
def mock_debate_result():
    """Create a mock debate result."""
    result = MagicMock()
    result.debate_id = "debate_test_123"
    result.task = "security audit process"
    result.proposals = [
        MagicMock(agent_type="claude"),
        MagicMock(agent_type="gpt4"),
    ]
    result.winner = "claude"
    result.consensus_reached = True
    result.rounds_used = 3
    result.confidence = 0.85
    result.critiques = []
    return result


@pytest.fixture
def mock_continuum_memory():
    """Create a mock ContinuumMemory."""
    mock = MagicMock()
    mock.retrieve.return_value = []
    mock.search_by_keyword.return_value = []
    return mock


@pytest.fixture
def mock_consensus_memory():
    """Create a mock ConsensusMemory."""
    mock = MagicMock()
    mock.search_by_topic = AsyncMock(return_value=[])
    mock._store = None
    return mock


@pytest.fixture
def mock_fact_store():
    """Create a mock FactStore."""
    mock = MagicMock()
    mock.query_facts.return_value = []
    return mock


@pytest.fixture
def mock_evidence_store():
    """Create a mock EvidenceStore."""
    mock = MagicMock()
    mock.search.return_value = []
    return mock


@pytest.fixture
def mock_critique_store():
    """Create a mock CritiqueStore."""
    mock = MagicMock()
    mock.search_patterns.return_value = []
    return mock

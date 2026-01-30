"""
Comprehensive tests for KnowledgeGraphStore.

Tests cover:
1. Graph construction and modification
2. Node operations (add, remove, update)
3. Edge operations (add, remove, update weights)
4. Cycle detection
5. Path finding for argument chains
6. Query operations
7. Performance with large graphs (1000+ nodes)
8. Serialization/deserialization
9. Concurrent access
10. Error handling edge cases

Run with: pytest tests/knowledge/mound/test_graph_store.py -v
"""

from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.mound.graph_store import (
    GraphLink,
    GraphTraversalResult,
    KnowledgeGraphStore,
    LineageNode,
)
from aragora.knowledge.mound.types import RelationshipType


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_graph.db"


@pytest.fixture
def graph_store(temp_db_path):
    """Create a KnowledgeGraphStore instance for testing."""
    return KnowledgeGraphStore(db_path=temp_db_path)


@pytest.fixture
def populated_graph_store(temp_db_path):
    """Create a graph store with pre-populated data."""
    store = KnowledgeGraphStore(db_path=temp_db_path)

    # Create synchronously for fixture setup
    def setup_data():
        with store.connection() as conn:
            # Add some test links
            for i in range(5):
                conn.execute(
                    """
                    INSERT INTO knowledge_links
                    (id, source_id, target_id, relationship, confidence,
                     created_by, created_at, tenant_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"link_{i}",
                        f"km_source_{i}",
                        f"km_target_{i}",
                        RelationshipType.SUPPORTS.value,
                        0.8,
                        "test_agent",
                        datetime.now().isoformat(),
                        "default",
                        "{}",
                    ),
                )

            # Add lineage chain: item_0 -> item_1 -> item_2
            for i in range(3):
                predecessor = f"km_lineage_{i - 1}" if i > 0 else None
                conn.execute(
                    """
                    INSERT INTO belief_lineage
                    (id, current_id, predecessor_id, supersession_reason,
                     debate_id, created_at, tenant_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"lin_{i}",
                        f"km_lineage_{i}",
                        predecessor,
                        "Updated with new evidence" if predecessor else None,
                        f"debate_{i}" if predecessor else None,
                        datetime.now().isoformat(),
                        "default",
                    ),
                )

    setup_data()
    return store


# ============================================================================
# Graph Construction and Modification Tests
# ============================================================================


class TestGraphConstruction:
    """Tests for graph construction and basic setup."""

    def test_store_initialization(self, temp_db_path):
        """Should initialize store with correct schema."""
        store = KnowledgeGraphStore(db_path=temp_db_path)

        # Verify tables exist
        tables = store.fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [t[0] for t in tables]

        assert "knowledge_links" in table_names
        assert "belief_lineage" in table_names
        assert "domain_taxonomy" in table_names
        assert "knowledge_links_archive" in table_names

    def test_store_with_custom_tenant(self, temp_db_path):
        """Should initialize with custom default tenant."""
        store = KnowledgeGraphStore(db_path=temp_db_path, default_tenant_id="custom_tenant")
        assert store._default_tenant_id == "custom_tenant"

    def test_schema_version(self, graph_store):
        """Should have correct schema version."""
        assert graph_store.SCHEMA_NAME == "knowledge_graph"
        assert graph_store.SCHEMA_VERSION == 1

    @pytest.mark.asyncio
    async def test_empty_graph_stats(self, graph_store):
        """Should return zero stats for empty graph."""
        stats = await graph_store.get_stats()

        assert stats["total_links"] == 0
        assert stats["by_relationship"] == {}
        assert stats["total_lineage_entries"] == 0
        assert stats["hub_nodes"] == []


# ============================================================================
# Node Operations Tests
# ============================================================================


class TestNodeOperations:
    """Tests for node-related operations."""

    @pytest.mark.asyncio
    async def test_add_link_creates_node_connection(self, graph_store):
        """Should create a link between two nodes."""
        link_id = await graph_store.add_link(
            source_id="km_node_a",
            target_id="km_node_b",
            relationship=RelationshipType.SUPPORTS,
            confidence=0.9,
            created_by="test_agent",
        )

        assert link_id.startswith("link_")

        # Verify link exists
        links = await graph_store.get_links("km_node_a", direction="outgoing")
        assert len(links) == 1
        assert links[0].source_id == "km_node_a"
        assert links[0].target_id == "km_node_b"

    @pytest.mark.asyncio
    async def test_get_links_for_nonexistent_node(self, graph_store):
        """Should return empty list for nonexistent node."""
        links = await graph_store.get_links("km_nonexistent")
        assert links == []

    @pytest.mark.asyncio
    async def test_delete_link_removes_node_connection(self, graph_store):
        """Should remove a link between nodes."""
        link_id = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
        )

        # Verify link exists
        links = await graph_store.get_links("km_a")
        assert len(links) == 1

        # Delete the link
        deleted = await graph_store.delete_link(link_id)
        assert deleted is True

        # Verify link is gone
        links = await graph_store.get_links("km_a")
        assert len(links) == 0

    @pytest.mark.asyncio
    async def test_node_with_multiple_links(self, graph_store):
        """Should handle nodes with multiple connections."""
        # Create a hub node with multiple connections
        for i in range(5):
            await graph_store.add_link(
                source_id="km_hub",
                target_id=f"km_target_{i}",
                relationship=RelationshipType.SUPPORTS,
            )

        links = await graph_store.get_links("km_hub", direction="outgoing")
        assert len(links) == 5


# ============================================================================
# Edge Operations Tests
# ============================================================================


class TestEdgeOperations:
    """Tests for edge (link) operations."""

    @pytest.mark.asyncio
    async def test_add_link_with_all_fields(self, graph_store):
        """Should create link with all metadata fields."""
        link_id = await graph_store.add_link(
            source_id="km_source",
            target_id="km_target",
            relationship=RelationshipType.CONTRADICTS,
            confidence=0.75,
            created_by="claude",
            tenant_id="tenant_1",
            metadata={"debate_id": "debate_123", "round": 2},
        )

        links = await graph_store.get_links("km_source", tenant_id="tenant_1")
        assert len(links) == 1
        link = links[0]

        assert link.relationship == RelationshipType.CONTRADICTS
        assert link.confidence == 0.75
        assert link.created_by == "claude"
        assert link.tenant_id == "tenant_1"
        assert link.metadata["debate_id"] == "debate_123"

    @pytest.mark.asyncio
    async def test_add_link_with_string_relationship(self, graph_store):
        """Should accept string relationship type."""
        link_id = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship="supports",  # String instead of enum
        )

        links = await graph_store.get_links("km_a")
        assert links[0].relationship == RelationshipType.SUPPORTS

    @pytest.mark.asyncio
    async def test_duplicate_link_returns_existing_id(self, graph_store):
        """Should return existing ID for duplicate link."""
        link_id1 = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
        )

        link_id2 = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
        )

        # Should return the existing link ID
        assert link_id1 == link_id2

    @pytest.mark.asyncio
    async def test_delete_link_with_archive(self, graph_store):
        """Should archive link when deleted."""
        link_id = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
        )

        await graph_store.delete_link(link_id, archive=True, reason="test_deletion")

        # Verify archived
        archived = graph_store.fetch_one(
            "SELECT archive_reason FROM knowledge_links_archive WHERE id = ?",
            (link_id,),
        )
        assert archived is not None
        assert archived[0] == "test_deletion"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_link(self, graph_store):
        """Should return False when deleting nonexistent link."""
        deleted = await graph_store.delete_link("nonexistent_link")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_links_by_relationship_type(self, graph_store):
        """Should filter links by relationship type."""
        await graph_store.add_link("km_a", "km_b", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_a", "km_c", RelationshipType.CONTRADICTS)
        await graph_store.add_link("km_a", "km_d", RelationshipType.ELABORATES)

        supports_links = await graph_store.get_links(
            "km_a",
            relationship_types=[RelationshipType.SUPPORTS],
        )
        assert len(supports_links) == 1
        assert supports_links[0].target_id == "km_b"

    @pytest.mark.asyncio
    async def test_get_links_incoming_direction(self, graph_store):
        """Should get only incoming links."""
        await graph_store.add_link("km_a", "km_center", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_b", "km_center", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_center", "km_c", RelationshipType.SUPPORTS)

        incoming = await graph_store.get_links("km_center", direction="incoming")
        assert len(incoming) == 2
        assert all(link.target_id == "km_center" for link in incoming)

    @pytest.mark.asyncio
    async def test_get_links_outgoing_direction(self, graph_store):
        """Should get only outgoing links."""
        await graph_store.add_link("km_center", "km_a", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_center", "km_b", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_c", "km_center", RelationshipType.SUPPORTS)

        outgoing = await graph_store.get_links("km_center", direction="outgoing")
        assert len(outgoing) == 2
        assert all(link.source_id == "km_center" for link in outgoing)


# ============================================================================
# Contradiction Detection Tests (Cycle-like pattern)
# ============================================================================


class TestContradictionDetection:
    """Tests for finding contradictions in the graph."""

    @pytest.mark.asyncio
    async def test_find_direct_contradiction(self, graph_store):
        """Should find direct contradictions."""
        await graph_store.add_link(
            "km_claim_a",
            "km_claim_b",
            RelationshipType.CONTRADICTS,
        )

        contradictions = await graph_store.find_contradictions("km_claim_a")
        assert "km_claim_b" in contradictions

    @pytest.mark.asyncio
    async def test_find_bidirectional_contradiction(self, graph_store):
        """Should find contradiction from either direction."""
        await graph_store.add_link(
            "km_claim_a",
            "km_claim_b",
            RelationshipType.CONTRADICTS,
        )

        # Should find from source
        contradictions_a = await graph_store.find_contradictions("km_claim_a")
        assert "km_claim_b" in contradictions_a

        # Should find from target
        contradictions_b = await graph_store.find_contradictions("km_claim_b")
        assert "km_claim_a" in contradictions_b

    @pytest.mark.asyncio
    async def test_no_contradictions(self, graph_store):
        """Should return empty list when no contradictions exist."""
        await graph_store.add_link(
            "km_a",
            "km_b",
            RelationshipType.SUPPORTS,
        )

        contradictions = await graph_store.find_contradictions("km_a")
        assert contradictions == []

    @pytest.mark.asyncio
    async def test_multiple_contradictions(self, graph_store):
        """Should find all contradicting items."""
        await graph_store.add_link("km_main", "km_contra_1", RelationshipType.CONTRADICTS)
        await graph_store.add_link("km_contra_2", "km_main", RelationshipType.CONTRADICTS)
        await graph_store.add_link("km_main", "km_support", RelationshipType.SUPPORTS)

        contradictions = await graph_store.find_contradictions("km_main")

        assert len(contradictions) == 2
        assert "km_contra_1" in contradictions
        assert "km_contra_2" in contradictions
        assert "km_support" not in contradictions


# ============================================================================
# Path Finding (Argument Chains) Tests
# ============================================================================


class TestPathFinding:
    """Tests for graph traversal and path finding."""

    @pytest.mark.asyncio
    async def test_traverse_simple_chain(self, graph_store):
        """Should traverse a simple chain of nodes."""
        # Create chain: A -> B -> C
        await graph_store.add_link("km_a", "km_b", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_b", "km_c", RelationshipType.SUPPORTS)

        result = await graph_store.traverse(
            start_id="km_a",
            max_depth=3,
            direction="outgoing",
        )

        assert "km_a" in result.nodes
        assert "km_b" in result.nodes
        assert "km_c" in result.nodes
        assert result.total_nodes == 3

    @pytest.mark.asyncio
    async def test_traverse_with_depth_limit(self, graph_store):
        """Should respect depth limit during traversal."""
        # Create chain: A -> B -> C -> D
        await graph_store.add_link("km_a", "km_b", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_b", "km_c", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_c", "km_d", RelationshipType.SUPPORTS)

        result = await graph_store.traverse(
            start_id="km_a",
            max_depth=2,
            direction="outgoing",
        )

        assert "km_a" in result.nodes
        assert "km_b" in result.nodes
        assert "km_c" in result.nodes
        assert "km_d" not in result.nodes  # Depth limit reached
        assert result.total_nodes == 3

    @pytest.mark.asyncio
    async def test_traverse_with_relationship_filter(self, graph_store):
        """Should filter by relationship type during traversal."""
        await graph_store.add_link("km_a", "km_b", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_a", "km_c", RelationshipType.CONTRADICTS)

        result = await graph_store.traverse(
            start_id="km_a",
            relationship_types=[RelationshipType.SUPPORTS],
            max_depth=2,
        )

        assert "km_b" in result.nodes
        assert "km_c" not in result.nodes

    @pytest.mark.asyncio
    async def test_traverse_bidirectional(self, graph_store):
        """Should traverse in both directions."""
        await graph_store.add_link("km_a", "km_center", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_center", "km_b", RelationshipType.SUPPORTS)

        result = await graph_store.traverse(
            start_id="km_center",
            max_depth=2,
            direction="both",
        )

        assert "km_a" in result.nodes
        assert "km_center" in result.nodes
        assert "km_b" in result.nodes

    @pytest.mark.asyncio
    async def test_traverse_handles_cycles(self, graph_store):
        """Should not get stuck in cycles during traversal."""
        # Create cycle: A -> B -> C -> A
        await graph_store.add_link("km_a", "km_b", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_b", "km_c", RelationshipType.SUPPORTS)
        await graph_store.add_link("km_c", "km_a", RelationshipType.SUPPORTS)

        result = await graph_store.traverse(
            start_id="km_a",
            max_depth=10,
            direction="outgoing",
        )

        # Should visit each node once
        assert result.total_nodes == 3
        assert "km_a" in result.nodes
        assert "km_b" in result.nodes
        assert "km_c" in result.nodes

    @pytest.mark.asyncio
    async def test_traverse_isolated_node(self, graph_store):
        """Should handle isolated nodes with no connections."""
        result = await graph_store.traverse(
            start_id="km_isolated",
            max_depth=3,
        )

        assert result.nodes == ["km_isolated"]
        assert result.edges == []
        assert result.total_nodes == 1
        assert result.total_edges == 0


# ============================================================================
# Query Operations Tests
# ============================================================================


class TestQueryOperations:
    """Tests for various query operations."""

    @pytest.mark.asyncio
    async def test_get_stats(self, populated_graph_store):
        """Should return accurate statistics."""
        stats = await populated_graph_store.get_stats()

        assert stats["total_links"] == 5
        assert RelationshipType.SUPPORTS.value in stats["by_relationship"]
        assert stats["by_relationship"][RelationshipType.SUPPORTS.value] == 5
        assert stats["total_lineage_entries"] == 3

    @pytest.mark.asyncio
    async def test_get_stats_with_tenant_filter(self, graph_store):
        """Should filter stats by tenant."""
        await graph_store.add_link("km_a", "km_b", RelationshipType.SUPPORTS, tenant_id="tenant_1")
        await graph_store.add_link("km_c", "km_d", RelationshipType.SUPPORTS, tenant_id="tenant_2")

        stats_1 = await graph_store.get_stats(tenant_id="tenant_1")
        assert stats_1["total_links"] == 1

        stats_2 = await graph_store.get_stats(tenant_id="tenant_2")
        assert stats_2["total_links"] == 1

    @pytest.mark.asyncio
    async def test_hub_nodes_identification(self, graph_store):
        """Should identify hub nodes with most connections."""
        # Create a hub with many connections
        for i in range(10):
            await graph_store.add_link("km_hub", f"km_target_{i}", RelationshipType.SUPPORTS)

        # Add some other nodes with fewer connections
        await graph_store.add_link("km_other", "km_single", RelationshipType.SUPPORTS)

        stats = await graph_store.get_stats()

        assert len(stats["hub_nodes"]) > 0
        assert stats["hub_nodes"][0]["node_id"] == "km_hub"
        assert stats["hub_nodes"][0]["connections"] == 10


# ============================================================================
# Belief Lineage Tests
# ============================================================================


class TestBeliefLineage:
    """Tests for belief lineage tracking."""

    @pytest.mark.asyncio
    async def test_add_lineage_origin(self, graph_store):
        """Should add origin lineage entry (no predecessor)."""
        lineage_id = await graph_store.add_lineage(
            current_id="km_origin",
            predecessor_id=None,
            supersession_reason=None,
        )

        assert lineage_id.startswith("lin_")

        lineage = await graph_store.get_lineage("km_origin")
        assert len(lineage) == 1
        assert lineage[0].predecessor_id is None

    @pytest.mark.asyncio
    async def test_add_lineage_with_predecessor(self, graph_store):
        """Should add lineage entry with predecessor."""
        await graph_store.add_lineage(current_id="km_v1")
        lineage_id = await graph_store.add_lineage(
            current_id="km_v2",
            predecessor_id="km_v1",
            supersession_reason="Updated based on new evidence",
            debate_id="debate_123",
        )

        lineage = await graph_store.get_lineage("km_v2", direction="predecessors")

        # Should have entry for both v2 and v1
        assert any(node.current_id == "km_v2" for node in lineage)
        assert any(node.supersession_reason == "Updated based on new evidence" for node in lineage)

    @pytest.mark.asyncio
    async def test_lineage_creates_supersedes_link(self, graph_store):
        """Should automatically create SUPERSEDES link for lineage."""
        await graph_store.add_lineage(current_id="km_old")
        await graph_store.add_lineage(
            current_id="km_new",
            predecessor_id="km_old",
        )

        # Check for SUPERSEDES link
        links = await graph_store.get_links(
            "km_new",
            relationship_types=[RelationshipType.SUPERSEDES],
            direction="outgoing",
        )

        assert len(links) == 1
        assert links[0].target_id == "km_old"

    @pytest.mark.asyncio
    async def test_get_lineage_predecessors(self, populated_graph_store):
        """Should get predecessor chain."""
        lineage = await populated_graph_store.get_lineage(
            "km_lineage_2",
            direction="predecessors",
        )

        # Should include entries for lineage_2 and lineage_1
        current_ids = [node.current_id for node in lineage]
        assert "km_lineage_2" in current_ids

    @pytest.mark.asyncio
    async def test_get_lineage_successors(self, populated_graph_store):
        """Should get successor chain."""
        lineage = await populated_graph_store.get_lineage(
            "km_lineage_0",
            direction="successors",
        )

        # Should include entries for lineage_1 and lineage_2
        current_ids = [node.current_id for node in lineage]
        assert "km_lineage_1" in current_ids
        assert "km_lineage_2" in current_ids

    @pytest.mark.asyncio
    async def test_get_lineage_chain(self, populated_graph_store):
        """Should get full lineage chain from oldest to newest."""
        chain = await populated_graph_store.get_lineage_chain("km_lineage_1")

        # Chain should start with oldest and end with newest
        assert "km_lineage_1" in chain

    @pytest.mark.asyncio
    async def test_lineage_tenant_isolation(self, graph_store):
        """Should isolate lineage by tenant."""
        await graph_store.add_lineage(
            current_id="km_item",
            tenant_id="tenant_a",
        )
        await graph_store.add_lineage(
            current_id="km_item",
            tenant_id="tenant_b",
        )

        lineage_a = await graph_store.get_lineage("km_item", tenant_id="tenant_a")
        lineage_b = await graph_store.get_lineage("km_item", tenant_id="tenant_b")

        assert len(lineage_a) == 1
        assert len(lineage_b) == 1
        assert lineage_a[0].tenant_id == "tenant_a"
        assert lineage_b[0].tenant_id == "tenant_b"


# ============================================================================
# Performance Tests (Large Graphs)
# ============================================================================


class TestPerformance:
    """Performance tests with large graphs."""

    @pytest.mark.asyncio
    async def test_large_graph_creation(self, graph_store):
        """Should efficiently create a graph with 1000+ nodes."""
        start_time = time.perf_counter()

        # Create 1000 nodes with connections
        for i in range(1000):
            await graph_store.add_link(
                source_id=f"km_node_{i}",
                target_id=f"km_node_{(i + 1) % 1000}",  # Circular connections
                relationship=RelationshipType.SUPPORTS,
            )

        elapsed = time.perf_counter() - start_time

        # Should complete in reasonable time
        assert elapsed < 30.0, f"Graph creation took {elapsed:.2f}s"

        stats = await graph_store.get_stats()
        assert stats["total_links"] == 1000

    @pytest.mark.asyncio
    async def test_large_graph_traversal(self, graph_store):
        """Should efficiently traverse a large graph."""
        # Create a tree structure with 1000 nodes
        for i in range(999):
            await graph_store.add_link(
                source_id=f"km_node_{i // 10}",  # Parent
                target_id=f"km_node_{i + 1}",  # Child
                relationship=RelationshipType.ELABORATES,
            )

        start_time = time.perf_counter()

        result = await graph_store.traverse(
            start_id="km_node_0",
            max_depth=5,
            direction="outgoing",
        )

        elapsed = time.perf_counter() - start_time

        assert elapsed < 5.0, f"Traversal took {elapsed:.2f}s"
        assert result.total_nodes > 0

    @pytest.mark.asyncio
    async def test_hub_node_with_many_connections(self, graph_store):
        """Should handle hub node with 500+ connections."""
        # Create a hub with 500 outgoing connections
        for i in range(500):
            await graph_store.add_link(
                source_id="km_mega_hub",
                target_id=f"km_leaf_{i}",
                relationship=RelationshipType.SUPPORTS,
            )

        start_time = time.perf_counter()
        links = await graph_store.get_links("km_mega_hub", direction="outgoing")
        elapsed = time.perf_counter() - start_time

        assert len(links) == 500
        assert elapsed < 1.0, f"Hub query took {elapsed:.2f}s"


# ============================================================================
# Serialization/Deserialization Tests
# ============================================================================


class TestSerialization:
    """Tests for data serialization and deserialization."""

    @pytest.mark.asyncio
    async def test_metadata_serialization(self, graph_store):
        """Should correctly serialize and deserialize metadata."""
        complex_metadata = {
            "debate_id": "debate_123",
            "agents": ["claude", "gpt-4"],
            "scores": {"claude": 0.85, "gpt-4": 0.78},
            "nested": {"level1": {"level2": "value"}},
        }

        await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
            metadata=complex_metadata,
        )

        links = await graph_store.get_links("km_a")
        assert links[0].metadata == complex_metadata

    @pytest.mark.asyncio
    async def test_datetime_handling(self, graph_store):
        """Should correctly store and retrieve datetime fields."""
        await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
        )

        links = await graph_store.get_links("km_a")
        assert isinstance(links[0].created_at, datetime)

    @pytest.mark.asyncio
    async def test_enum_serialization(self, graph_store):
        """Should correctly serialize and deserialize relationship enums."""
        for rel_type in RelationshipType:
            await graph_store.add_link(
                source_id=f"km_source_{rel_type.value}",
                target_id=f"km_target_{rel_type.value}",
                relationship=rel_type,
            )

        for rel_type in RelationshipType:
            links = await graph_store.get_links(
                f"km_source_{rel_type.value}",
                relationship_types=[rel_type],
            )
            assert len(links) == 1
            assert links[0].relationship == rel_type


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_link_additions(self, graph_store):
        """Should handle concurrent link additions."""

        async def add_links(prefix: str, count: int):
            for i in range(count):
                await graph_store.add_link(
                    source_id=f"km_{prefix}_source_{i}",
                    target_id=f"km_{prefix}_target_{i}",
                    relationship=RelationshipType.SUPPORTS,
                )

        # Run 5 concurrent batches of 20 links each
        tasks = [add_links(f"batch_{i}", 20) for i in range(5)]
        await asyncio.gather(*tasks)

        stats = await graph_store.get_stats()
        assert stats["total_links"] == 100

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, graph_store):
        """Should handle concurrent reads and writes."""
        write_count = 0
        read_count = 0

        async def writer():
            nonlocal write_count
            for i in range(50):
                await graph_store.add_link(
                    source_id=f"km_write_{i}",
                    target_id=f"km_target_{i}",
                    relationship=RelationshipType.SUPPORTS,
                )
                write_count += 1

        async def reader():
            nonlocal read_count
            for _ in range(50):
                await graph_store.get_stats()
                read_count += 1

        await asyncio.gather(writer(), reader(), reader())

        assert write_count == 50
        assert read_count == 100

    @pytest.mark.asyncio
    async def test_concurrent_traversals(self, graph_store):
        """Should handle concurrent graph traversals."""
        # Create a graph first
        for i in range(100):
            await graph_store.add_link(
                source_id=f"km_node_{i}",
                target_id=f"km_node_{(i + 1) % 100}",
                relationship=RelationshipType.SUPPORTS,
            )

        # Run multiple concurrent traversals
        async def traverse(start_idx: int):
            return await graph_store.traverse(
                start_id=f"km_node_{start_idx}",
                max_depth=3,
            )

        tasks = [traverse(i * 10) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(isinstance(r, GraphTraversalResult) for r in results)
        assert all(r.total_nodes > 0 for r in results)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_relationship_type_stored_as_string(self, graph_store):
        """Should store invalid relationship type as string (validated at query time)."""
        # The store accepts any string - validation happens when deserializing
        link_id = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship="invalid_relationship",
        )

        assert link_id.startswith("link_")

        # When getting links, deserialization will fail with invalid enum
        with pytest.raises(ValueError):
            await graph_store.get_links("km_a")

    @pytest.mark.asyncio
    async def test_empty_node_id(self, graph_store):
        """Should handle empty node IDs."""
        # Empty source should still work (no constraint on ID format)
        link_id = await graph_store.add_link(
            source_id="",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
        )
        assert link_id.startswith("link_")

    @pytest.mark.asyncio
    async def test_very_long_metadata(self, graph_store):
        """Should handle large metadata objects."""
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        link_id = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
            metadata=large_metadata,
        )

        links = await graph_store.get_links("km_a")
        assert links[0].metadata == large_metadata

    @pytest.mark.asyncio
    async def test_self_referencing_link(self, graph_store):
        """Should allow self-referencing links."""
        link_id = await graph_store.add_link(
            source_id="km_self",
            target_id="km_self",
            relationship=RelationshipType.RELATED_TO,
        )

        links = await graph_store.get_links("km_self")
        # Should appear in both directions
        assert len(links) >= 1

    @pytest.mark.asyncio
    async def test_negative_confidence(self, graph_store):
        """Should handle negative confidence values."""
        link_id = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
            confidence=-0.5,  # Negative value
        )

        links = await graph_store.get_links("km_a")
        assert links[0].confidence == -0.5

    @pytest.mark.asyncio
    async def test_confidence_above_one(self, graph_store):
        """Should handle confidence values above 1."""
        link_id = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
            confidence=1.5,
        )

        links = await graph_store.get_links("km_a")
        assert links[0].confidence == 1.5

    @pytest.mark.asyncio
    async def test_unicode_in_ids(self, graph_store):
        """Should handle unicode characters in node IDs."""
        link_id = await graph_store.add_link(
            source_id="km_cafe",
            target_id="km_niho",
            relationship=RelationshipType.SUPPORTS,
        )

        links = await graph_store.get_links("km_cafe")
        assert len(links) == 1
        assert links[0].target_id == "km_niho"

    @pytest.mark.asyncio
    async def test_special_characters_in_metadata(self, graph_store):
        """Should handle special characters in metadata."""
        special_metadata = {
            "quotes": 'He said "hello"',
            "newlines": "line1\nline2",
            "tabs": "col1\tcol2",
            "backslash": "path\\to\\file",
            "null_char": "before\x00after",
        }

        link_id = await graph_store.add_link(
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
            metadata=special_metadata,
        )

        links = await graph_store.get_links("km_a")
        # Metadata should be preserved (null char may be handled differently)
        assert "quotes" in links[0].metadata


# ============================================================================
# GraphLink and LineageNode Dataclass Tests
# ============================================================================


class TestDataclasses:
    """Tests for dataclass behavior."""

    def test_graph_link_creation(self):
        """Should create GraphLink with all fields."""
        link = GraphLink(
            id="link_123",
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
            confidence=0.85,
            created_by="claude",
            created_at=datetime.now(),
            tenant_id="default",
            metadata={"key": "value"},
        )

        assert link.id == "link_123"
        assert link.confidence == 0.85
        assert link.metadata["key"] == "value"

    def test_graph_link_default_metadata(self):
        """Should have empty dict as default metadata."""
        link = GraphLink(
            id="link_123",
            source_id="km_a",
            target_id="km_b",
            relationship=RelationshipType.SUPPORTS,
            confidence=0.85,
            created_by=None,
            created_at=datetime.now(),
            tenant_id="default",
        )

        assert link.metadata == {}

    def test_lineage_node_creation(self):
        """Should create LineageNode with all fields."""
        node = LineageNode(
            id="lin_123",
            current_id="km_v2",
            predecessor_id="km_v1",
            supersession_reason="New evidence",
            debate_id="debate_456",
            created_at=datetime.now(),
            tenant_id="default",
        )

        assert node.id == "lin_123"
        assert node.predecessor_id == "km_v1"
        assert node.supersession_reason == "New evidence"

    def test_graph_traversal_result_creation(self):
        """Should create GraphTraversalResult with all fields."""
        result = GraphTraversalResult(
            nodes=["km_a", "km_b", "km_c"],
            edges=[],
            root_id="km_a",
            depth=2,
            total_nodes=3,
            total_edges=0,
        )

        assert result.total_nodes == 3
        assert result.root_id == "km_a"


# ============================================================================
# Multi-Tenant Isolation Tests
# ============================================================================


class TestMultiTenantIsolation:
    """Tests for tenant isolation."""

    @pytest.mark.asyncio
    async def test_links_isolated_by_tenant(self, graph_store):
        """Should isolate links by tenant."""
        await graph_store.add_link("km_a", "km_b", RelationshipType.SUPPORTS, tenant_id="tenant_1")
        await graph_store.add_link("km_a", "km_c", RelationshipType.SUPPORTS, tenant_id="tenant_2")

        links_1 = await graph_store.get_links("km_a", tenant_id="tenant_1")
        links_2 = await graph_store.get_links("km_a", tenant_id="tenant_2")

        assert len(links_1) == 1
        assert links_1[0].target_id == "km_b"

        assert len(links_2) == 1
        assert links_2[0].target_id == "km_c"

    @pytest.mark.asyncio
    async def test_contradictions_isolated_by_tenant(self, graph_store):
        """Should find contradictions only within tenant."""
        await graph_store.add_link(
            "km_claim", "km_contra_1", RelationshipType.CONTRADICTS, tenant_id="tenant_1"
        )
        await graph_store.add_link(
            "km_claim", "km_contra_2", RelationshipType.CONTRADICTS, tenant_id="tenant_2"
        )

        contradictions_1 = await graph_store.find_contradictions("km_claim", tenant_id="tenant_1")
        contradictions_2 = await graph_store.find_contradictions("km_claim", tenant_id="tenant_2")

        assert contradictions_1 == ["km_contra_1"]
        assert contradictions_2 == ["km_contra_2"]

    @pytest.mark.asyncio
    async def test_traversal_isolated_by_tenant(self, graph_store):
        """Should traverse only within tenant."""
        # Create different graphs for different tenants
        await graph_store.add_link(
            "km_start", "km_a1", RelationshipType.SUPPORTS, tenant_id="tenant_1"
        )
        await graph_store.add_link(
            "km_start", "km_b1", RelationshipType.SUPPORTS, tenant_id="tenant_2"
        )

        result_1 = await graph_store.traverse("km_start", tenant_id="tenant_1")
        result_2 = await graph_store.traverse("km_start", tenant_id="tenant_2")

        assert "km_a1" in result_1.nodes
        assert "km_b1" not in result_1.nodes

        assert "km_b1" in result_2.nodes
        assert "km_a1" not in result_2.nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

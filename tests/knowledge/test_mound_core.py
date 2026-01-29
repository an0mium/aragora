"""Comprehensive unit tests for aragora/knowledge/mound_core.py.

Tests cover:
- Helper functions (_to_iso_string, _to_enum_value)
- ProvenanceChain creation, transformation, serialization
- KnowledgeNode properties, promotion/demotion, from_fact conversion
- KnowledgeRelationship lifecycle
- KnowledgeQueryResult structure
- KnowledgeMoundMetaStore: CRUD, relationships, queries, provenance, stats, deletion
- KnowledgeMound: initialization, deduplication, semantic search, graph traversal,
  provenance queries, deletion, export, error handling
"""

import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound_core import (
    KnowledgeMound,
    KnowledgeMoundMetaStore,
    KnowledgeNode,
    KnowledgeQueryResult,
    KnowledgeRelationship,
    ProvenanceChain,
    ProvenanceType,
    _to_enum_value,
    _to_iso_string,
)
from aragora.knowledge.types import Fact, ValidationStatus
from aragora.memory.tier_manager import MemoryTier


# ============================================================
# Helper functions
# ============================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_to_iso_string_with_none(self):
        """Returns None when given None."""
        assert _to_iso_string(None) is None

    def test_to_iso_string_with_string(self):
        """Returns string unchanged."""
        iso = "2025-01-15T12:00:00"
        assert _to_iso_string(iso) == iso

    def test_to_iso_string_with_datetime(self):
        """Converts datetime to ISO string."""
        dt = datetime(2025, 1, 15, 12, 0, 0)
        assert _to_iso_string(dt) == "2025-01-15T12:00:00"

    def test_to_iso_string_with_arbitrary_object(self):
        """Falls back to str() for unknown types."""
        assert _to_iso_string(42) == "42"

    def test_to_enum_value_with_none(self):
        """Returns None when given None."""
        assert _to_enum_value(None) is None

    def test_to_enum_value_with_string(self):
        """Returns string unchanged."""
        assert _to_enum_value("fast") == "fast"

    def test_to_enum_value_with_enum(self):
        """Extracts .value from enum instances."""
        assert _to_enum_value(MemoryTier.FAST) == "fast"
        assert _to_enum_value(ProvenanceType.DEBATE) == "debate"

    def test_to_enum_value_with_arbitrary_object(self):
        """Falls back to str() for objects without .value."""
        assert _to_enum_value(123) == "123"


# ============================================================
# ProvenanceChain
# ============================================================


class TestProvenanceChainExtended:
    """Extended tests for ProvenanceChain beyond basic create/serialize."""

    def test_multiple_transformations_preserve_order(self):
        """Transformations maintain insertion order."""
        prov = ProvenanceChain(
            source_type=ProvenanceType.DOCUMENT,
            source_id="doc_1",
        )
        prov.add_transformation("extract", agent_id="agent_a")
        prov.add_transformation("refine", agent_id="agent_b")
        prov.add_transformation("validate", agent_id="agent_c")

        assert len(prov.transformations) == 3
        assert prov.transformations[0]["type"] == "extract"
        assert prov.transformations[1]["type"] == "refine"
        assert prov.transformations[2]["type"] == "validate"

    def test_transformation_records_timestamp(self):
        """Each transformation includes a timestamp."""
        prov = ProvenanceChain(
            source_type=ProvenanceType.USER,
            source_id="user_1",
        )
        prov.add_transformation("edit", details={"field": "content"})

        assert "timestamp" in prov.transformations[0]
        # Should be a valid ISO string
        datetime.fromisoformat(prov.transformations[0]["timestamp"])

    def test_from_dict_with_missing_optional_fields(self):
        """Handles missing optional fields gracefully."""
        data = {
            "source_type": "user",
            "source_id": "user_1",
        }
        prov = ProvenanceChain.from_dict(data)
        assert prov.agent_id is None
        assert prov.debate_id is None
        assert prov.document_id is None
        assert prov.user_id is None
        assert prov.transformations == []

    def test_from_dict_with_non_string_created_at(self):
        """Handles non-string created_at by falling back to now."""
        data = {
            "source_type": "agent",
            "source_id": "agent_1",
            "created_at": 12345,  # Not a string
        }
        prov = ProvenanceChain.from_dict(data)
        # Should default to a recent datetime (now)
        assert isinstance(prov.created_at, datetime)

    def test_all_provenance_types_roundtrip(self):
        """Every ProvenanceType serializes and deserializes correctly."""
        for pt in ProvenanceType:
            prov = ProvenanceChain(source_type=pt, source_id="test")
            restored = ProvenanceChain.from_dict(prov.to_dict())
            assert restored.source_type == pt


# ============================================================
# KnowledgeNode
# ============================================================


class TestKnowledgeNodeExtended:
    """Extended tests for KnowledgeNode beyond basic property checks."""

    def test_auto_generated_id_is_unique(self):
        """Two nodes created without explicit ID have different IDs."""
        n1 = KnowledgeNode(content="A")
        n2 = KnowledgeNode(content="B")
        assert n1.id != n2.id

    def test_content_hash_deterministic(self):
        """Content hash matches manual SHA-256 computation."""
        content = "Deterministic hash test"
        node = KnowledgeNode(content=content)
        expected = hashlib.sha256(content.encode()).hexdigest()[:32]
        assert node.content_hash == expected

    def test_stability_score_inverse_of_surprise(self):
        """stability_score = 1 - surprise_score."""
        node = KnowledgeNode(content="X", surprise_score=0.3)
        assert node.stability_score == pytest.approx(0.7)

    def test_should_promote_requires_both_conditions(self):
        """Promotion requires surprise > 0.7 AND update_count > 5."""
        # High surprise but low update count -> no promotion
        node = KnowledgeNode(
            content="X",
            tier=MemoryTier.SLOW,
            surprise_score=0.9,
            update_count=3,
        )
        assert not node.should_promote()

    def test_should_demote_requires_both_conditions(self):
        """Demotion requires stability > 0.8 AND consolidation > 0.5."""
        # High stability but low consolidation -> no demotion
        node = KnowledgeNode(
            content="X",
            tier=MemoryTier.FAST,
            surprise_score=0.1,
            consolidation_score=0.2,
        )
        assert not node.should_demote()

    def test_from_fact_conversion(self):
        """KnowledgeNode.from_fact populates fields correctly from a Fact."""
        fact = Fact(
            id="fact_42",
            statement="The sky is blue",
            confidence=0.92,
            source_documents=["doc_alpha"],
            workspace_id="ws_science",
            validation_status=ValidationStatus.MAJORITY_AGREED,
            consensus_proof_id="proof_7",
            topics=["nature", "physics"],
            metadata={"source": "observation"},
        )
        node = KnowledgeNode.from_fact(fact)

        assert node.id == "kn_fact_42"
        assert node.node_type == "fact"
        assert node.content == "The sky is blue"
        assert node.confidence == 0.92
        assert node.workspace_id == "ws_science"
        assert node.validation_status == ValidationStatus.MAJORITY_AGREED
        assert node.consensus_proof_id == "proof_7"
        assert node.topics == ["nature", "physics"]
        assert node.provenance is not None
        assert node.provenance.source_type == ProvenanceType.DOCUMENT
        assert node.provenance.document_id == "doc_alpha"

    def test_from_fact_with_empty_source_documents(self):
        """from_fact handles facts with no source documents."""
        fact = Fact(
            id="fact_empty",
            statement="No sources",
            source_documents=[],
        )
        node = KnowledgeNode.from_fact(fact, workspace_id="ws_default")
        assert node.provenance.source_id == ""
        assert node.provenance.document_id is None
        assert node.workspace_id == "ws_default"

    def test_from_dict_with_defaults(self):
        """from_dict fills defaults for missing optional fields."""
        data = {
            "id": "kn_minimal",
            "content": "Minimal node",
        }
        node = KnowledgeNode.from_dict(data)
        assert node.node_type == "fact"
        assert node.confidence == 0.5
        assert node.tier == MemoryTier.SLOW
        assert node.workspace_id == ""
        assert node.provenance is None


# ============================================================
# KnowledgeRelationship
# ============================================================


class TestKnowledgeRelationshipExtended:
    """Extended tests for KnowledgeRelationship."""

    def test_auto_generated_id_prefix(self):
        """Auto-generated IDs start with kr_."""
        rel = KnowledgeRelationship(from_node_id="a", to_node_id="b")
        assert rel.id.startswith("kr_")

    def test_from_dict_defaults(self):
        """from_dict uses defaults for missing optional fields."""
        data = {
            "id": "kr_test",
            "from_node_id": "a",
            "to_node_id": "b",
        }
        rel = KnowledgeRelationship.from_dict(data)
        assert rel.relationship_type == "related_to"
        assert rel.strength == 1.0
        assert rel.created_by == ""
        assert rel.metadata == {}


# ============================================================
# KnowledgeQueryResult
# ============================================================


class TestKnowledgeQueryResult:
    """Tests for KnowledgeQueryResult dataclass."""

    def test_query_result_fields(self):
        """Verify all fields are populated."""
        nodes = [KnowledgeNode(content="A"), KnowledgeNode(content="B")]
        result = KnowledgeQueryResult(
            nodes=nodes,
            total_count=2,
            query="test query",
            processing_time_ms=42,
            metadata={"engine": "sqlite"},
        )
        assert len(result.nodes) == 2
        assert result.total_count == 2
        assert result.query == "test query"
        assert result.processing_time_ms == 42
        assert result.metadata["engine"] == "sqlite"

    def test_query_result_defaults(self):
        """Default processing_time_ms is 0 and metadata is empty dict."""
        result = KnowledgeQueryResult(nodes=[], total_count=0, query="")
        assert result.processing_time_ms == 0
        assert result.metadata == {}


# ============================================================
# KnowledgeMoundMetaStore
# ============================================================


class TestMetaStoreDeleteAndProvenance:
    """Tests for MetaStore deletion and provenance queries."""

    @pytest.fixture
    def store(self):
        """Create a temporary MetaStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_mound_core.db"
            yield KnowledgeMoundMetaStore(db_path)

    def test_delete_node_returns_false_for_missing(self, store):
        """Deleting a nonexistent node returns False."""
        assert store.delete_node("kn_nonexistent") is False

    def test_delete_node_removes_relationships(self, store):
        """Deleting a node also removes its relationships."""
        n1 = KnowledgeNode(content="Node 1", workspace_id="ws")
        n2 = KnowledgeNode(content="Node 2", workspace_id="ws")
        store.save_node(n1)
        store.save_node(n2)
        rel = KnowledgeRelationship(
            from_node_id=n1.id, to_node_id=n2.id, relationship_type="supports"
        )
        store.save_relationship(rel)

        store.delete_node(n1.id)

        # Relationship should be gone from both directions
        assert store.get_relationships(n2.id, direction="incoming") == []

    def test_query_by_provenance_source_type(self, store):
        """Can filter nodes by provenance source type."""
        node = KnowledgeNode(
            content="Debate result",
            workspace_id="ws",
            provenance=ProvenanceChain(
                source_type=ProvenanceType.DEBATE,
                source_id="debate_1",
                debate_id="debate_1",
            ),
        )
        store.save_node(node)

        ids = store.query_by_provenance(source_type="debate", workspace_id="ws")
        assert node.id in ids

        ids_agent = store.query_by_provenance(source_type="agent", workspace_id="ws")
        assert node.id not in ids_agent

    def test_get_relationships_incoming(self, store):
        """Can query incoming relationships."""
        n1 = KnowledgeNode(content="Source", workspace_id="ws")
        n2 = KnowledgeNode(content="Target", workspace_id="ws")
        store.save_node(n1)
        store.save_node(n2)
        rel = KnowledgeRelationship(
            from_node_id=n1.id, to_node_id=n2.id, relationship_type="supports"
        )
        store.save_relationship(rel)

        incoming = store.get_relationships(n2.id, direction="incoming")
        assert len(incoming) == 1
        assert incoming[0].from_node_id == n1.id

    def test_get_relationships_both_directions(self, store):
        """Direction='both' returns outgoing and incoming."""
        n1 = KnowledgeNode(content="A", workspace_id="ws")
        n2 = KnowledgeNode(content="B", workspace_id="ws")
        n3 = KnowledgeNode(content="C", workspace_id="ws")
        store.save_node(n1)
        store.save_node(n2)
        store.save_node(n3)

        # n1 -> n2, n3 -> n1
        store.save_relationship(
            KnowledgeRelationship(
                from_node_id=n1.id, to_node_id=n2.id, relationship_type="supports"
            )
        )
        store.save_relationship(
            KnowledgeRelationship(
                from_node_id=n3.id, to_node_id=n1.id, relationship_type="contradicts"
            )
        )

        both = store.get_relationships(n1.id, direction="both")
        assert len(both) == 2

    def test_get_relationships_filtered_by_type(self, store):
        """Can filter relationships by type."""
        n1 = KnowledgeNode(content="A", workspace_id="ws")
        n2 = KnowledgeNode(content="B", workspace_id="ws")
        n3 = KnowledgeNode(content="C", workspace_id="ws")
        store.save_node(n1)
        store.save_node(n2)
        store.save_node(n3)

        store.save_relationship(
            KnowledgeRelationship(
                from_node_id=n1.id, to_node_id=n2.id, relationship_type="supports"
            )
        )
        store.save_relationship(
            KnowledgeRelationship(
                from_node_id=n1.id, to_node_id=n3.id, relationship_type="contradicts"
            )
        )

        supports_only = store.get_relationships(
            n1.id, relationship_type="supports", direction="outgoing"
        )
        assert len(supports_only) == 1
        assert supports_only[0].to_node_id == n2.id

    def test_query_nodes_by_topics(self, store):
        """Can filter nodes by topics."""
        n1 = KnowledgeNode(content="Security best practice", workspace_id="ws", topics=["security"])
        n2 = KnowledgeNode(content="Performance tip", workspace_id="ws", topics=["performance"])
        store.save_node(n1)
        store.save_node(n2)

        security_nodes = store.query_nodes(workspace_id="ws", topics=["security"])
        assert len(security_nodes) == 1
        assert security_nodes[0].id == n1.id

    def test_query_nodes_by_validation_status(self, store):
        """Can filter nodes by validation status."""
        n1 = KnowledgeNode(
            content="Verified",
            workspace_id="ws",
            validation_status=ValidationStatus.FORMALLY_PROVEN,
        )
        n2 = KnowledgeNode(
            content="Unverified",
            workspace_id="ws",
            validation_status=ValidationStatus.UNVERIFIED,
        )
        store.save_node(n1)
        store.save_node(n2)

        proven = store.query_nodes(
            workspace_id="ws", validation_status=ValidationStatus.FORMALLY_PROVEN
        )
        assert len(proven) == 1
        assert proven[0].content == "Verified"

    def test_get_node_returns_none_for_missing(self, store):
        """get_node returns None for nonexistent ID."""
        assert store.get_node("kn_does_not_exist") is None

    def test_save_node_upsert_replaces(self, store):
        """Saving a node with same ID replaces the existing one."""
        node = KnowledgeNode(id="kn_fixed", content="Original", workspace_id="ws", confidence=0.5)
        store.save_node(node)

        node.content = "Updated"
        node.confidence = 0.9
        store.save_node(node)

        retrieved = store.get_node("kn_fixed")
        assert retrieved.content == "Updated"
        assert retrieved.confidence == 0.9


# ============================================================
# KnowledgeMound (async integration)
# ============================================================


class TestKnowledgeMoundCore:
    """Core async tests for KnowledgeMound."""

    @pytest.fixture
    def mound(self):
        """Create a temporary KnowledgeMound."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_core_mound.db"
            m = KnowledgeMound(workspace_id="test_ws", db_path=db_path)
            asyncio.run(m.initialize())
            yield m
            asyncio.run(m.close())

    def test_ensure_initialized_raises_before_init(self):
        """Calling methods before initialize() raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "uninit.db"
            m = KnowledgeMound(workspace_id="x", db_path=db_path)
            with pytest.raises(RuntimeError, match="not initialized"):
                m._ensure_initialized()

    @pytest.mark.asyncio
    async def test_double_initialize_is_noop(self, mound):
        """Calling initialize() twice does not error."""
        await mound.initialize()  # second call
        assert mound._initialized is True

    @pytest.mark.asyncio
    async def test_delete_node_returns_true(self, mound):
        """Deleting an existing node returns True."""
        node = KnowledgeNode(content="To be deleted")
        node_id = await mound.add_node(node)
        result = await mound.delete_node(node_id)
        assert result is True

        # Verify it is gone
        assert await mound.get_node(node_id) is None

    @pytest.mark.asyncio
    async def test_delete_node_returns_false_for_missing(self, mound):
        """Deleting a nonexistent node returns False."""
        result = await mound.delete_node("kn_ghost")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_node_sets_workspace_automatically(self, mound):
        """Nodes without workspace_id inherit the mound's workspace."""
        node = KnowledgeNode(content="Auto workspace")
        await mound.add_node(node)
        retrieved = await mound.get_node(node.id)
        assert retrieved.workspace_id == "test_ws"

    @pytest.mark.asyncio
    async def test_deduplication_merges_confidence(self, mound):
        """Duplicate content merges confidence via weighted average."""
        n1 = KnowledgeNode(content="Same content", confidence=1.0)
        n2 = KnowledgeNode(content="Same content", confidence=0.5)

        await mound.add_node(n1, deduplicate=True)
        await mound.add_node(n2, deduplicate=True)

        retrieved = await mound.get_node(n1.id)
        # 1.0 * 0.7 + 0.5 * 0.3 = 0.85
        assert retrieved.confidence == pytest.approx(0.85, abs=0.01)

    @pytest.mark.asyncio
    async def test_deduplication_disabled(self, mound):
        """With deduplicate=False, two nodes with same content are both stored."""
        n1 = KnowledgeNode(content="Dup content")
        n2 = KnowledgeNode(content="Dup content")

        id1 = await mound.add_node(n1, deduplicate=False)
        id2 = await mound.add_node(n2, deduplicate=False)

        # They should have different IDs since dedup is off
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_query_by_provenance(self, mound):
        """Can query nodes by provenance attributes."""
        node = KnowledgeNode(
            content="Workflow output",
            provenance=ProvenanceChain(
                source_type=ProvenanceType.INFERENCE,
                source_id="wf_42",
            ),
        )
        await mound.add_node(node)

        results = await mound.query_by_provenance(source_type="inference")
        assert len(results) >= 1
        assert any(n.id == node.id for n in results)

    @pytest.mark.asyncio
    async def test_get_relationships_via_mound(self, mound):
        """KnowledgeMound.get_relationships delegates correctly."""
        n1 = KnowledgeNode(content="A")
        n2 = KnowledgeNode(content="B")
        await mound.add_node(n1)
        await mound.add_node(n2)

        await mound.add_relationship(n1.id, n2.id, "contradicts", strength=0.6)

        rels = await mound.get_relationships(n1.id, relationship_type="contradicts")
        assert len(rels) >= 1
        assert rels[0].to_node_id == n2.id
        assert rels[0].strength == 0.6

    @pytest.mark.asyncio
    async def test_semantic_query_keyword_fallback(self, mound):
        """Without vector store, semantic search uses keyword matching."""
        await mound.add_node(KnowledgeNode(content="Python is a programming language"))
        await mound.add_node(KnowledgeNode(content="Cats are furry animals"))

        result = await mound.query_semantic("Python programming")
        # "Python" and "programming" overlap with first node
        assert result.total_count >= 1
        assert result.processing_time_ms >= 0
        contents = [n.content for n in result.nodes]
        assert any("Python" in c for c in contents)

    @pytest.mark.asyncio
    async def test_export_graph_d3_no_start_node(self, mound):
        """D3 export without start_node returns all nodes."""
        n1 = KnowledgeNode(content="Node A", confidence=0.8)
        n2 = KnowledgeNode(content="Node B", confidence=0.6)
        await mound.add_node(n1)
        await mound.add_node(n2)
        await mound.add_relationship(n1.id, n2.id, "supports")

        d3 = await mound.export_graph_d3()
        assert len(d3["nodes"]) == 2
        assert len(d3["links"]) == 1
        assert d3["links"][0]["source"] == n1.id
        assert d3["links"][0]["target"] == n2.id

    @pytest.mark.asyncio
    async def test_export_graph_graphml(self, mound):
        """GraphML export produces valid XML structure."""
        node = KnowledgeNode(content="Test & <special> chars", confidence=0.7)
        await mound.add_node(node)

        xml = await mound.export_graph_graphml()
        assert xml.startswith('<?xml version="1.0"')
        assert "<graphml" in xml
        assert "</graphml>" in xml
        # Special characters should be escaped
        assert "&amp;" in xml
        assert "&lt;special&gt;" in xml

    @pytest.mark.asyncio
    async def test_close_resets_initialized(self, mound):
        """Closing the mound sets _initialized to False."""
        assert mound._initialized is True
        await mound.close()
        assert mound._initialized is False

    @pytest.mark.asyncio
    async def test_get_stats_returns_structure(self, mound):
        """Stats include expected keys."""
        await mound.add_node(KnowledgeNode(content="Stats test", node_type="evidence"))
        stats = await mound.get_stats()
        assert "total_nodes" in stats
        assert "by_type" in stats
        assert "by_tier" in stats
        assert "average_confidence" in stats
        assert "total_relationships" in stats
        assert stats["total_nodes"] >= 1

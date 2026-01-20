"""Tests for Knowledge Mound unified knowledge storage."""

import asyncio
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from aragora.knowledge.mound_core import (
    KnowledgeMound,
    KnowledgeMoundMetaStore,
    KnowledgeNode,
    KnowledgeRelationship,
    KnowledgeQueryResult,
    ProvenanceChain,
    ProvenanceType,
)
from aragora.knowledge.types import ValidationStatus
from aragora.memory.tier_manager import MemoryTier


class TestKnowledgeNode:
    """Test KnowledgeNode data structure."""

    def test_create_node_with_defaults(self):
        """Test creating a node with default values."""
        node = KnowledgeNode(
            content="API keys should never be committed to version control",
        )

        assert node.id.startswith("kn_")
        assert node.node_type == "fact"
        assert node.content == "API keys should never be committed to version control"
        assert node.confidence == 0.5
        assert node.tier == MemoryTier.SLOW
        assert node.validation_status == ValidationStatus.UNVERIFIED
        assert node.surprise_score == 0.0
        assert node.update_count == 1

    def test_create_node_with_custom_values(self):
        """Test creating a node with custom values."""
        provenance = ProvenanceChain(
            source_type=ProvenanceType.DEBATE,
            source_id="debate_123",
            debate_id="debate_123",
        )

        node = KnowledgeNode(
            id="kn_custom_123",
            node_type="consensus",
            content="Multi-factor authentication is recommended",
            confidence=0.95,
            provenance=provenance,
            tier=MemoryTier.GLACIAL,
            workspace_id="ws_security",
            validation_status=ValidationStatus.MAJORITY_AGREED,
            topics=["security", "authentication"],
        )

        assert node.id == "kn_custom_123"
        assert node.node_type == "consensus"
        assert node.confidence == 0.95
        assert node.tier == MemoryTier.GLACIAL
        assert node.workspace_id == "ws_security"
        assert node.provenance.source_type == ProvenanceType.DEBATE
        assert "security" in node.topics

    def test_content_hash(self):
        """Test content hash generation."""
        node1 = KnowledgeNode(content="Same content")
        node2 = KnowledgeNode(content="Same content")
        node3 = KnowledgeNode(content="Different content")

        assert node1.content_hash == node2.content_hash
        assert node1.content_hash != node3.content_hash

    def test_is_verified(self):
        """Test is_verified property."""
        node_unverified = KnowledgeNode(
            content="Test",
            validation_status=ValidationStatus.UNVERIFIED,
        )
        node_contested = KnowledgeNode(
            content="Test",
            validation_status=ValidationStatus.CONTESTED,
        )
        node_majority = KnowledgeNode(
            content="Test",
            validation_status=ValidationStatus.MAJORITY_AGREED,
        )
        node_byzantine = KnowledgeNode(
            content="Test",
            validation_status=ValidationStatus.BYZANTINE_AGREED,
        )
        node_formal = KnowledgeNode(
            content="Test",
            validation_status=ValidationStatus.FORMALLY_PROVEN,
        )

        assert not node_unverified.is_verified
        assert not node_contested.is_verified
        assert node_majority.is_verified
        assert node_byzantine.is_verified
        assert node_formal.is_verified

    def test_should_promote(self):
        """Test promotion logic."""
        # High surprise in medium tier -> should promote
        node_promote = KnowledgeNode(
            content="Test",
            tier=MemoryTier.MEDIUM,
            surprise_score=0.8,
            update_count=10,
        )
        assert node_promote.should_promote()

        # Already in fast tier -> should not promote
        node_fast = KnowledgeNode(
            content="Test",
            tier=MemoryTier.FAST,
            surprise_score=0.9,
        )
        assert not node_fast.should_promote()

        # Low surprise -> should not promote
        node_low_surprise = KnowledgeNode(
            content="Test",
            tier=MemoryTier.MEDIUM,
            surprise_score=0.3,
        )
        assert not node_low_surprise.should_promote()

    def test_should_demote(self):
        """Test demotion logic."""
        # High stability, high consolidation -> should demote
        node_demote = KnowledgeNode(
            content="Test",
            tier=MemoryTier.FAST,
            surprise_score=0.1,  # Low surprise = high stability
            consolidation_score=0.7,
        )
        assert node_demote.should_demote()

        # Already in glacial -> should not demote
        node_glacial = KnowledgeNode(
            content="Test",
            tier=MemoryTier.GLACIAL,
            surprise_score=0.1,
            consolidation_score=0.9,
        )
        assert not node_glacial.should_demote()

    def test_to_dict_from_dict(self):
        """Test serialization/deserialization."""
        provenance = ProvenanceChain(
            source_type=ProvenanceType.USER,
            source_id="user_123",
            user_id="user_123",
        )

        node = KnowledgeNode(
            id="kn_test",
            node_type="fact",
            content="Test content",
            confidence=0.8,
            provenance=provenance,
            tier=MemoryTier.MEDIUM,
            workspace_id="ws_test",
            topics=["topic1", "topic2"],
            metadata={"key": "value"},
        )

        node_dict = node.to_dict()
        restored = KnowledgeNode.from_dict(node_dict)

        assert restored.id == node.id
        assert restored.node_type == node.node_type
        assert restored.content == node.content
        assert restored.confidence == node.confidence
        assert restored.tier == node.tier
        assert restored.workspace_id == node.workspace_id
        assert restored.topics == node.topics
        assert restored.metadata == node.metadata


class TestProvenanceChain:
    """Test ProvenanceChain data structure."""

    def test_create_provenance(self):
        """Test creating provenance."""
        prov = ProvenanceChain(
            source_type=ProvenanceType.DOCUMENT,
            source_id="doc_123",
            document_id="doc_123",
        )

        assert prov.source_type == ProvenanceType.DOCUMENT
        assert prov.source_id == "doc_123"
        assert prov.document_id == "doc_123"
        assert len(prov.transformations) == 0

    def test_add_transformation(self):
        """Test adding transformations."""
        prov = ProvenanceChain(
            source_type=ProvenanceType.DOCUMENT,
            source_id="doc_123",
        )

        prov.add_transformation(
            transform_type="extraction",
            agent_id="claude",
            details={"method": "NER"},
        )

        assert len(prov.transformations) == 1
        assert prov.transformations[0]["type"] == "extraction"
        assert prov.transformations[0]["agent_id"] == "claude"
        assert prov.transformations[0]["details"]["method"] == "NER"

    def test_to_dict_from_dict(self):
        """Test serialization/deserialization."""
        prov = ProvenanceChain(
            source_type=ProvenanceType.DEBATE,
            source_id="debate_123",
            agent_id="claude",
            debate_id="debate_123",
        )
        prov.add_transformation("refinement", agent_id="gpt-4")

        prov_dict = prov.to_dict()
        restored = ProvenanceChain.from_dict(prov_dict)

        assert restored.source_type == prov.source_type
        assert restored.source_id == prov.source_id
        assert restored.agent_id == prov.agent_id
        assert restored.debate_id == prov.debate_id
        assert len(restored.transformations) == 1


class TestKnowledgeMoundMetaStore:
    """Test KnowledgeMoundMetaStore SQLite operations."""

    @pytest.fixture
    def store(self):
        """Create temporary store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_mound.db"
            yield KnowledgeMoundMetaStore(db_path)

    def test_save_and_get_node(self, store):
        """Test saving and retrieving a node."""
        node = KnowledgeNode(
            node_type="fact",
            content="Test fact content",
            confidence=0.8,
            workspace_id="ws_test",
            topics=["test", "example"],
        )

        node_id = store.save_node(node)
        assert node_id == node.id

        retrieved = store.get_node(node.id)
        assert retrieved is not None
        assert retrieved.id == node.id
        assert retrieved.content == node.content
        assert retrieved.confidence == node.confidence
        assert "test" in retrieved.topics
        assert "example" in retrieved.topics

    def test_save_node_with_provenance(self, store):
        """Test saving node with provenance."""
        provenance = ProvenanceChain(
            source_type=ProvenanceType.AGENT,
            source_id="agent_claude",
            agent_id="claude",
        )

        node = KnowledgeNode(
            node_type="claim",
            content="Agent-generated claim",
            provenance=provenance,
            workspace_id="ws_test",
        )

        store.save_node(node)
        retrieved = store.get_node(node.id)

        assert retrieved.provenance is not None
        assert retrieved.provenance.source_type == ProvenanceType.AGENT
        assert retrieved.provenance.agent_id == "claude"

    def test_save_and_get_relationship(self, store):
        """Test saving and retrieving relationships."""
        node1 = KnowledgeNode(content="Premise", workspace_id="ws_test")
        node2 = KnowledgeNode(content="Conclusion", workspace_id="ws_test")
        store.save_node(node1)
        store.save_node(node2)

        rel = KnowledgeRelationship(
            from_node_id=node1.id,
            to_node_id=node2.id,
            relationship_type="supports",
            strength=0.9,
        )
        store.save_relationship(rel)

        relationships = store.get_relationships(node1.id, direction="outgoing")
        assert len(relationships) == 1
        assert relationships[0].to_node_id == node2.id
        assert relationships[0].relationship_type == "supports"

    def test_query_nodes_by_type(self, store):
        """Test querying nodes by type."""
        store.save_node(KnowledgeNode(node_type="fact", content="Fact 1", workspace_id="ws_test"))
        store.save_node(KnowledgeNode(node_type="fact", content="Fact 2", workspace_id="ws_test"))
        store.save_node(KnowledgeNode(node_type="claim", content="Claim 1", workspace_id="ws_test"))

        facts = store.query_nodes(workspace_id="ws_test", node_types=["fact"])
        assert len(facts) == 2

        claims = store.query_nodes(workspace_id="ws_test", node_types=["claim"])
        assert len(claims) == 1

    def test_query_nodes_by_confidence(self, store):
        """Test querying nodes by confidence threshold."""
        store.save_node(KnowledgeNode(content="Low", confidence=0.3, workspace_id="ws_test"))
        store.save_node(KnowledgeNode(content="Medium", confidence=0.6, workspace_id="ws_test"))
        store.save_node(KnowledgeNode(content="High", confidence=0.9, workspace_id="ws_test"))

        high_confidence = store.query_nodes(workspace_id="ws_test", min_confidence=0.7)
        assert len(high_confidence) == 1
        assert high_confidence[0].content == "High"

    def test_query_nodes_by_tier(self, store):
        """Test querying nodes by tier."""
        store.save_node(KnowledgeNode(content="Fast", tier=MemoryTier.FAST, workspace_id="ws_test"))
        store.save_node(KnowledgeNode(content="Slow", tier=MemoryTier.SLOW, workspace_id="ws_test"))
        store.save_node(KnowledgeNode(content="Glacial", tier=MemoryTier.GLACIAL, workspace_id="ws_test"))

        fast_nodes = store.query_nodes(workspace_id="ws_test", tier=MemoryTier.FAST)
        assert len(fast_nodes) == 1
        assert fast_nodes[0].content == "Fast"

    def test_find_by_content_hash(self, store):
        """Test finding node by content hash."""
        node = KnowledgeNode(content="Unique content", workspace_id="ws_test")
        store.save_node(node)

        # Same content -> find existing
        duplicate = KnowledgeNode(content="Unique content", workspace_id="ws_test")
        found = store.find_by_content_hash(duplicate.content_hash, "ws_test")
        assert found is not None
        assert found.id == node.id

        # Different workspace -> not found
        not_found = store.find_by_content_hash(duplicate.content_hash, "ws_other")
        assert not_found is None

    def test_get_stats(self, store):
        """Test getting statistics."""
        store.save_node(KnowledgeNode(
            node_type="fact",
            content="Fact 1",
            confidence=0.8,
            tier=MemoryTier.FAST,
            workspace_id="ws_test",
        ))
        store.save_node(KnowledgeNode(
            node_type="fact",
            content="Fact 2",
            confidence=0.9,
            tier=MemoryTier.SLOW,
            workspace_id="ws_test",
        ))
        store.save_node(KnowledgeNode(
            node_type="claim",
            content="Claim 1",
            confidence=0.7,
            tier=MemoryTier.SLOW,
            workspace_id="ws_test",
        ))

        stats = store.get_stats("ws_test")

        assert stats["total_nodes"] == 3
        assert stats["by_type"]["fact"] == 2
        assert stats["by_type"]["claim"] == 1
        assert stats["by_tier"]["fast"] == 1
        assert stats["by_tier"]["slow"] == 2
        assert stats["average_confidence"] == pytest.approx(0.8, rel=0.01)


class TestKnowledgeMound:
    """Test KnowledgeMound unified interface."""

    @pytest.fixture
    def mound(self):
        """Create temporary mound for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_mound.db"
            m = KnowledgeMound(workspace_id="test", db_path=db_path)
            asyncio.run(m.initialize())
            yield m
            asyncio.run(m.close())

    @pytest.mark.asyncio
    async def test_add_and_get_node(self, mound):
        """Test adding and retrieving a node."""
        node = KnowledgeNode(
            node_type="fact",
            content="Test knowledge content",
            confidence=0.85,
        )

        node_id = await mound.add_node(node)
        assert node_id == node.id

        retrieved = await mound.get_node(node_id)
        assert retrieved is not None
        assert retrieved.content == node.content
        assert retrieved.confidence == node.confidence
        assert retrieved.workspace_id == "test"

    @pytest.mark.asyncio
    async def test_deduplication(self, mound):
        """Test that duplicate content is deduplicated."""
        node1 = KnowledgeNode(content="Duplicate content", confidence=0.7)
        node2 = KnowledgeNode(content="Duplicate content", confidence=0.9)

        id1 = await mound.add_node(node1, deduplicate=True)
        id2 = await mound.add_node(node2, deduplicate=True)

        assert id1 == id2  # Should be same node

        # Confidence should be updated (weighted average)
        retrieved = await mound.get_node(id1)
        assert retrieved.update_count == 2

    @pytest.mark.asyncio
    async def test_add_relationship(self, mound):
        """Test adding relationships between nodes."""
        premise = KnowledgeNode(content="Premise content")
        conclusion = KnowledgeNode(content="Conclusion content")

        await mound.add_node(premise)
        await mound.add_node(conclusion)

        rel_id = await mound.add_relationship(
            from_node_id=premise.id,
            to_node_id=conclusion.id,
            relationship_type="supports",
            strength=0.9,
        )

        assert rel_id.startswith("kr_")

    @pytest.mark.asyncio
    async def test_query_semantic(self, mound):
        """Test semantic query."""
        await mound.add_node(KnowledgeNode(
            content="API keys should be stored securely",
            topics=["security"],
        ))
        await mound.add_node(KnowledgeNode(
            content="Use environment variables for secrets",
            topics=["security"],
        ))
        await mound.add_node(KnowledgeNode(
            content="The weather is nice today",
            topics=["weather"],
        ))

        # Query should return security-related nodes
        result = await mound.query_semantic("security best practices", limit=10)

        assert result.total_count >= 2
        # Note: Without actual embeddings, this is keyword-based
        security_contents = [n.content for n in result.nodes]
        assert any("security" in c.lower() or "secret" in c.lower() for c in security_contents)

    @pytest.mark.asyncio
    async def test_query_graph(self, mound):
        """Test graph traversal."""
        root = KnowledgeNode(content="Root node")
        child1 = KnowledgeNode(content="Child 1")
        child2 = KnowledgeNode(content="Child 2")
        grandchild = KnowledgeNode(content="Grandchild")

        await mound.add_node(root)
        await mound.add_node(child1)
        await mound.add_node(child2)
        await mound.add_node(grandchild)

        await mound.add_relationship(root.id, child1.id, "supports")
        await mound.add_relationship(root.id, child2.id, "supports")
        await mound.add_relationship(child1.id, grandchild.id, "supports")

        # Traverse from root with depth 2
        nodes = await mound.query_graph(root.id, "supports", depth=2)

        # Should include root, children, and grandchild
        node_contents = [n.content for n in nodes]
        assert "Root node" in node_contents
        assert "Child 1" in node_contents
        assert "Child 2" in node_contents
        assert "Grandchild" in node_contents

    @pytest.mark.asyncio
    async def test_query_nodes_filtered(self, mound):
        """Test filtered node queries."""
        await mound.add_node(KnowledgeNode(
            node_type="fact",
            content="High confidence fact",
            confidence=0.95,
            tier=MemoryTier.FAST,
        ))
        await mound.add_node(KnowledgeNode(
            node_type="claim",
            content="Low confidence claim",
            confidence=0.3,
            tier=MemoryTier.SLOW,
        ))

        # Filter by type
        facts = await mound.query_nodes(node_types=["fact"])
        assert len(facts) == 1
        assert facts[0].node_type == "fact"

        # Filter by confidence
        high_conf = await mound.query_nodes(min_confidence=0.9)
        assert len(high_conf) == 1
        assert high_conf[0].confidence >= 0.9

        # Filter by tier
        fast_tier = await mound.query_nodes(tier=MemoryTier.FAST)
        assert len(fast_tier) == 1
        assert fast_tier[0].tier == MemoryTier.FAST

    @pytest.mark.asyncio
    async def test_get_stats(self, mound):
        """Test getting mound statistics."""
        await mound.add_node(KnowledgeNode(content="Node 1"))
        await mound.add_node(KnowledgeNode(content="Node 2"))

        stats = await mound.get_stats()

        assert stats["total_nodes"] == 2


class TestKnowledgeRelationship:
    """Test KnowledgeRelationship data structure."""

    def test_create_relationship(self):
        """Test creating a relationship."""
        rel = KnowledgeRelationship(
            from_node_id="kn_1",
            to_node_id="kn_2",
            relationship_type="supports",
            strength=0.8,
            created_by="claude",
        )

        assert rel.id.startswith("kr_")
        assert rel.from_node_id == "kn_1"
        assert rel.to_node_id == "kn_2"
        assert rel.relationship_type == "supports"
        assert rel.strength == 0.8
        assert rel.created_by == "claude"

    def test_to_dict_from_dict(self):
        """Test serialization/deserialization."""
        rel = KnowledgeRelationship(
            id="kr_test",
            from_node_id="kn_1",
            to_node_id="kn_2",
            relationship_type="contradicts",
            strength=0.7,
            metadata={"reason": "conflicting evidence"},
        )

        rel_dict = rel.to_dict()
        restored = KnowledgeRelationship.from_dict(rel_dict)

        assert restored.id == rel.id
        assert restored.from_node_id == rel.from_node_id
        assert restored.to_node_id == rel.to_node_id
        assert restored.relationship_type == rel.relationship_type
        assert restored.strength == rel.strength
        assert restored.metadata == rel.metadata

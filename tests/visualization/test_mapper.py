"""
Tests for argument cartographer / mapper.

Tests cover:
- NodeType enum
- EdgeRelation enum
- ArgumentNode dataclass
- ArgumentEdge dataclass
- ArgumentCartographer class
"""

import time

import pytest

from aragora.visualization.mapper import (
    ArgumentCartographer,
    ArgumentEdge,
    ArgumentNode,
    EdgeRelation,
    NodeType,
)


# ============================================================================
# NodeType Tests
# ============================================================================


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_values(self):
        """Test all node types are defined."""
        expected = [
            "proposal",
            "critique",
            "evidence",
            "concession",
            "rebuttal",
            "vote",
            "consensus",
        ]
        assert all(NodeType(v) for v in expected)

    def test_value_access(self):
        """Test accessing enum values."""
        assert NodeType.PROPOSAL.value == "proposal"
        assert NodeType.CRITIQUE.value == "critique"
        assert NodeType.CONSENSUS.value == "consensus"


# ============================================================================
# EdgeRelation Tests
# ============================================================================


class TestEdgeRelation:
    """Tests for EdgeRelation enum."""

    def test_all_values(self):
        """Test all edge relations are defined."""
        expected = ["supports", "refutes", "modifies", "responds_to", "concedes_to"]
        assert all(EdgeRelation(v) for v in expected)

    def test_value_access(self):
        """Test accessing enum values."""
        assert EdgeRelation.SUPPORTS.value == "supports"
        assert EdgeRelation.REFUTES.value == "refutes"
        assert EdgeRelation.CONCEDES_TO.value == "concedes_to"


# ============================================================================
# ArgumentNode Tests
# ============================================================================


class TestArgumentNode:
    """Tests for ArgumentNode dataclass."""

    def test_creation(self):
        """Test basic creation."""
        node = ArgumentNode(
            id="node-1",
            agent="claude",
            node_type=NodeType.PROPOSAL,
            summary="This is a proposal",
            round_num=1,
            timestamp=time.time(),
        )
        assert node.id == "node-1"
        assert node.agent == "claude"
        assert node.node_type == NodeType.PROPOSAL
        assert node.summary == "This is a proposal"
        assert node.round_num == 1

    def test_default_values(self):
        """Test default values."""
        node = ArgumentNode(
            id="n1",
            agent="gpt",
            node_type=NodeType.CRITIQUE,
            summary="Test",
            round_num=1,
            timestamp=0.0,
        )
        assert node.full_content is None
        assert node.metadata == {}

    def test_with_full_content(self):
        """Test with full content."""
        full_text = "This is a very long argument that explains everything in detail."
        node = ArgumentNode(
            id="n1",
            agent="claude",
            node_type=NodeType.EVIDENCE,
            summary="This is a very...",
            round_num=2,
            timestamp=time.time(),
            full_content=full_text,
        )
        assert node.full_content == full_text

    def test_with_metadata(self):
        """Test with metadata."""
        node = ArgumentNode(
            id="n1",
            agent="gemini",
            node_type=NodeType.VOTE,
            summary="Votes: approve",
            round_num=3,
            timestamp=time.time(),
            metadata={"vote_value": "approve", "confidence": 0.9},
        )
        assert node.metadata["vote_value"] == "approve"
        assert node.metadata["confidence"] == 0.9

    def test_to_dict(self):
        """Test serialization to dictionary."""
        node = ArgumentNode(
            id="node-1",
            agent="claude",
            node_type=NodeType.PROPOSAL,
            summary="Test proposal",
            round_num=1,
            timestamp=1234567890.0,
            metadata={"key": "value"},
        )
        result = node.to_dict()

        assert result["id"] == "node-1"
        assert result["agent"] == "claude"
        assert result["node_type"] == "proposal"
        assert result["summary"] == "Test proposal"
        assert result["round_num"] == 1
        assert result["timestamp"] == 1234567890.0
        assert result["metadata"] == {"key": "value"}


# ============================================================================
# ArgumentEdge Tests
# ============================================================================


class TestArgumentEdge:
    """Tests for ArgumentEdge dataclass."""

    def test_creation(self):
        """Test basic creation."""
        edge = ArgumentEdge(
            source_id="node-1",
            target_id="node-2",
            relation=EdgeRelation.SUPPORTS,
        )
        assert edge.source_id == "node-1"
        assert edge.target_id == "node-2"
        assert edge.relation == EdgeRelation.SUPPORTS

    def test_default_values(self):
        """Test default values."""
        edge = ArgumentEdge(
            source_id="a",
            target_id="b",
            relation=EdgeRelation.REFUTES,
        )
        assert edge.weight == 1.0
        assert edge.metadata == {}

    def test_with_weight(self):
        """Test with custom weight."""
        edge = ArgumentEdge(
            source_id="a",
            target_id="b",
            relation=EdgeRelation.MODIFIES,
            weight=0.5,
        )
        assert edge.weight == 0.5

    def test_with_metadata(self):
        """Test with metadata."""
        edge = ArgumentEdge(
            source_id="a",
            target_id="b",
            relation=EdgeRelation.RESPONDS_TO,
            metadata={"critique_text": "This is a critique"},
        )
        assert edge.metadata["critique_text"] == "This is a critique"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        edge = ArgumentEdge(
            source_id="node-1",
            target_id="node-2",
            relation=EdgeRelation.SUPPORTS,
            weight=0.8,
            metadata={"reason": "supports main point"},
        )
        result = edge.to_dict()

        assert result["source_id"] == "node-1"
        assert result["target_id"] == "node-2"
        assert result["relation"] == "supports"
        assert result["weight"] == 0.8
        assert result["metadata"]["reason"] == "supports main point"


# ============================================================================
# ArgumentCartographer Tests
# ============================================================================


class TestArgumentCartographer:
    """Tests for ArgumentCartographer class."""

    @pytest.fixture
    def cartographer(self):
        """Create a fresh cartographer."""
        return ArgumentCartographer()

    def test_creation(self, cartographer):
        """Test basic creation."""
        assert cartographer.nodes == {}
        assert cartographer.edges == []
        assert cartographer.debate_id is None
        assert cartographer.topic is None

    def test_set_debate_context(self, cartographer):
        """Test setting debate context."""
        cartographer.set_debate_context("debate-123", "Should AI be regulated?")

        assert cartographer.debate_id == "debate-123"
        assert cartographer.topic == "Should AI be regulated?"

    def test_update_from_message_creates_node(self, cartographer):
        """Test message creates a node."""
        node_id = cartographer.update_from_message(
            agent="claude",
            content="I propose we focus on safety.",
            role="proposal",
            round_num=1,
        )

        assert node_id in cartographer.nodes
        node = cartographer.nodes[node_id]
        assert node.agent == "claude"
        assert node.round_num == 1

    def test_update_from_message_truncates_summary(self, cartographer):
        """Test long content is truncated in summary."""
        long_content = "x" * 200
        node_id = cartographer.update_from_message(
            agent="claude",
            content=long_content,
            role="proposal",
            round_num=1,
        )

        node = cartographer.nodes[node_id]
        assert len(node.summary) <= 103  # 100 + "..."
        assert node.full_content == long_content

    def test_update_from_message_with_metadata(self, cartographer):
        """Test message with metadata."""
        node_id = cartographer.update_from_message(
            agent="gpt",
            content="Test message",
            role="critique",
            round_num=2,
            metadata={"confidence": 0.95},
        )

        node = cartographer.nodes[node_id]
        assert node.metadata["confidence"] == 0.95

    def test_multiple_messages_creates_multiple_nodes(self, cartographer):
        """Test multiple messages create multiple nodes."""
        cartographer.update_from_message("claude", "Message 1", "proposal", 1)
        cartographer.update_from_message("gpt", "Message 2", "critique", 1)
        cartographer.update_from_message("gemini", "Message 3", "rebuttal", 1)

        assert len(cartographer.nodes) == 3

    def test_update_from_critique(self, cartographer):
        """Test critique creates an edge."""
        # First create nodes for both agents
        cartographer.update_from_message("claude", "Proposal", "proposal", 1)
        cartographer.update_from_message("gpt", "Critique", "critique", 1)

        edge_id = cartographer.update_from_critique(
            critic_agent="gpt",
            target_agent="claude",
            severity=0.8,
            round_num=1,
            critique_text="This is problematic",
        )

        assert edge_id is not None
        assert len(cartographer.edges) >= 1

    def test_update_from_critique_severity_high(self, cartographer):
        """Test high severity creates refutes edge."""
        cartographer.update_from_message("claude", "Proposal", "proposal", 1)
        cartographer.update_from_message("gpt", "Strong critique", "critique", 1)

        cartographer.update_from_critique("gpt", "claude", severity=0.9, round_num=1)

        # Find the edge with REFUTES relation
        refutes_edges = [e for e in cartographer.edges if e.relation == EdgeRelation.REFUTES]
        assert len(refutes_edges) >= 1

    def test_update_from_critique_severity_medium(self, cartographer):
        """Test medium severity creates modifies edge."""
        cartographer.update_from_message("claude", "Proposal", "proposal", 1)
        cartographer.update_from_message("gpt", "Moderate critique", "critique", 1)

        cartographer.update_from_critique("gpt", "claude", severity=0.5, round_num=1)

        modifies_edges = [e for e in cartographer.edges if e.relation == EdgeRelation.MODIFIES]
        assert len(modifies_edges) >= 1

    def test_update_from_critique_no_nodes(self, cartographer):
        """Test critique with no existing nodes returns None."""
        edge_id = cartographer.update_from_critique(
            "unknown1", "unknown2", severity=0.5, round_num=1
        )
        assert edge_id is None

    def test_update_from_vote(self, cartographer):
        """Test vote creates a vote node."""
        node_id = cartographer.update_from_vote(
            agent="claude",
            vote_value="approve",
            round_num=3,
        )

        assert node_id in cartographer.nodes
        node = cartographer.nodes[node_id]
        assert node.node_type == NodeType.VOTE
        assert "approve" in node.summary
        assert node.metadata["vote_value"] == "approve"

    def test_get_statistics(self, cartographer):
        """Test getting graph statistics."""
        cartographer.set_debate_context("d1", "Test topic")
        cartographer.update_from_message("claude", "Proposal", "proposal", 1)
        cartographer.update_from_message("gpt", "Critique", "critique", 1)
        cartographer.update_from_message("claude", "Response", "rebuttal", 2)

        stats = cartographer.get_statistics()

        assert stats["node_count"] == 3
        assert stats["rounds"] == 2
        assert "claude" in stats["agents"]
        assert "gpt" in stats["agents"]

    def test_export_json(self, cartographer):
        """Test JSON export."""
        cartographer.set_debate_context("d1", "Test topic")
        cartographer.update_from_message("claude", "Test proposal", "proposal", 1)

        import json

        json_str = cartographer.export_json()
        data = json.loads(json_str)

        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert data["metadata"]["node_count"] >= 1

    def test_export_mermaid(self, cartographer):
        """Test Mermaid export."""
        cartographer.set_debate_context("d1", "Test topic")
        cartographer.update_from_message("claude", "Proposal text", "proposal", 1)
        cartographer.update_from_message("gpt", "Critique text", "critique", 1)

        mermaid = cartographer.export_mermaid()

        assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower()

    def test_reset_via_new_instance(self, cartographer):
        """Test that a new instance starts fresh."""
        cartographer.set_debate_context("d1", "Test")
        cartographer.update_from_message("claude", "Test", "proposal", 1)

        assert len(cartographer.nodes) > 0

        # Create fresh instance to verify clean state
        new_cart = ArgumentCartographer()

        assert len(new_cart.nodes) == 0
        assert len(new_cart.edges) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestMapperIntegration:
    """Integration tests for the mapper module."""

    def test_full_debate_workflow(self):
        """Test complete debate workflow."""
        cart = ArgumentCartographer()
        cart.set_debate_context("debate-001", "AI Safety Discussion")

        # Round 1: Initial proposals
        cart.update_from_message("claude", "We should prioritize safety.", "proposal", 1)
        cart.update_from_message("gpt", "Speed is more important.", "proposal", 1)

        # Round 1: Critiques
        cart.update_from_message("claude", "Speed without safety is risky.", "critique", 1)
        cart.update_from_critique("claude", "gpt", 0.7, 1, "This ignores risks")

        # Round 2: Rebuttals
        cart.update_from_message("gpt", "We can have both.", "rebuttal", 2)

        # Round 3: Votes
        cart.update_from_vote("claude", "partial_agree", 3)
        cart.update_from_vote("gpt", "agree", 3)

        # Verify graph structure
        stats = cart.get_statistics()
        assert stats["node_count"] >= 5
        assert stats["rounds"] == 3
        assert len(stats["agents"]) == 2

        # Verify export works
        json_out = cart.export_json()
        assert "nodes" in json_out

        mermaid_out = cart.export_mermaid()
        assert len(mermaid_out) > 0

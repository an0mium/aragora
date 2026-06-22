"""Tests for aragora.visualization.mapper module.

Comprehensive tests for ArgumentCartographer, ArgumentNode, ArgumentEdge,
and related enums and functionality.
"""

import json
import pytest
import time
from unittest.mock import patch

from aragora.visualization.mapper import (
    NodeType,
    EdgeRelation,
    ArgumentNode,
    ArgumentEdge,
    ArgumentCartographer,
)


# =============================================================================
# NodeType Tests
# =============================================================================


class TestNodeType:
    """Tests for NodeType enum."""

    def test_proposal_value(self):
        """Test PROPOSAL has correct value."""
        assert NodeType.PROPOSAL.value == "proposal"

    def test_critique_value(self):
        """Test CRITIQUE has correct value."""
        assert NodeType.CRITIQUE.value == "critique"

    def test_evidence_value(self):
        """Test EVIDENCE has correct value."""
        assert NodeType.EVIDENCE.value == "evidence"

    def test_concession_value(self):
        """Test CONCESSION has correct value."""
        assert NodeType.CONCESSION.value == "concession"

    def test_rebuttal_value(self):
        """Test REBUTTAL has correct value."""
        assert NodeType.REBUTTAL.value == "rebuttal"

    def test_vote_value(self):
        """Test VOTE has correct value."""
        assert NodeType.VOTE.value == "vote"

    def test_consensus_value(self):
        """Test CONSENSUS has correct value."""
        assert NodeType.CONSENSUS.value == "consensus"


# =============================================================================
# EdgeRelation Tests
# =============================================================================


class TestEdgeRelation:
    """Tests for EdgeRelation enum."""

    def test_supports_value(self):
        """Test SUPPORTS has correct value."""
        assert EdgeRelation.SUPPORTS.value == "supports"

    def test_refutes_value(self):
        """Test REFUTES has correct value."""
        assert EdgeRelation.REFUTES.value == "refutes"

    def test_modifies_value(self):
        """Test MODIFIES has correct value."""
        assert EdgeRelation.MODIFIES.value == "modifies"

    def test_responds_to_value(self):
        """Test RESPONDS_TO has correct value."""
        assert EdgeRelation.RESPONDS_TO.value == "responds_to"

    def test_concedes_to_value(self):
        """Test CONCEDES_TO has correct value."""
        assert EdgeRelation.CONCEDES_TO.value == "concedes_to"


# =============================================================================
# ArgumentNode Tests
# =============================================================================


class TestArgumentNode:
    """Tests for ArgumentNode dataclass."""

    def test_create_minimal(self):
        """Test creating node with required fields."""
        node = ArgumentNode(
            id="node-1",
            agent="agent1",
            node_type=NodeType.PROPOSAL,
            summary="Test summary",
            round_num=1,
            timestamp=1234567890.0,
        )
        assert node.id == "node-1"
        assert node.agent == "agent1"
        assert node.node_type == NodeType.PROPOSAL
        assert node.summary == "Test summary"
        assert node.round_num == 1
        assert node.timestamp == 1234567890.0
        assert node.full_content is None
        assert node.metadata == {}

    def test_create_with_optional_fields(self):
        """Test creating node with all fields."""
        node = ArgumentNode(
            id="node-2",
            agent="agent2",
            node_type=NodeType.CRITIQUE,
            summary="Critique summary",
            round_num=2,
            timestamp=1234567891.0,
            full_content="Full critique content here",
            metadata={"severity": 0.8},
        )
        assert node.full_content == "Full critique content here"
        assert node.metadata == {"severity": 0.8}

    def test_to_dict(self):
        """Test to_dict serialization."""
        node = ArgumentNode(
            id="node-3",
            agent="agent3",
            node_type=NodeType.EVIDENCE,
            summary="Evidence",
            round_num=3,
            timestamp=1234567892.0,
            metadata={"source": "paper"},
        )
        d = node.to_dict()

        assert d["id"] == "node-3"
        assert d["agent"] == "agent3"
        assert d["node_type"] == "evidence"  # Enum value, not enum
        assert d["summary"] == "Evidence"
        assert d["round_num"] == 3
        assert d["timestamp"] == 1234567892.0
        assert d["metadata"] == {"source": "paper"}
        # Note: full_content not included in to_dict by default

    def test_to_dict_does_not_include_full_content(self):
        """Test to_dict excludes full_content by default."""
        node = ArgumentNode(
            id="n",
            agent="a",
            node_type=NodeType.PROPOSAL,
            summary="s",
            round_num=1,
            timestamp=0.0,
            full_content="Full content",
        )
        d = node.to_dict()
        assert "full_content" not in d


# =============================================================================
# ArgumentEdge Tests
# =============================================================================


class TestArgumentEdge:
    """Tests for ArgumentEdge dataclass."""

    def test_create_minimal(self):
        """Test creating edge with required fields."""
        edge = ArgumentEdge(
            source_id="node-1",
            target_id="node-2",
            relation=EdgeRelation.SUPPORTS,
        )
        assert edge.source_id == "node-1"
        assert edge.target_id == "node-2"
        assert edge.relation == EdgeRelation.SUPPORTS
        assert edge.weight == 1.0  # Default
        assert edge.metadata == {}  # Default

    def test_create_with_optional_fields(self):
        """Test creating edge with all fields."""
        edge = ArgumentEdge(
            source_id="node-3",
            target_id="node-4",
            relation=EdgeRelation.REFUTES,
            weight=0.8,
            metadata={"reason": "contradiction"},
        )
        assert edge.weight == 0.8
        assert edge.metadata == {"reason": "contradiction"}

    def test_to_dict(self):
        """Test to_dict serialization."""
        edge = ArgumentEdge(
            source_id="src",
            target_id="tgt",
            relation=EdgeRelation.MODIFIES,
            weight=0.5,
            metadata={"info": "test"},
        )
        d = edge.to_dict()

        assert d["source_id"] == "src"
        assert d["target_id"] == "tgt"
        assert d["relation"] == "modifies"  # Enum value
        assert d["weight"] == 0.5
        assert d["metadata"] == {"info": "test"}


# =============================================================================
# ArgumentCartographer Basic Tests
# =============================================================================


class TestArgumentCartographer:
    """Tests for ArgumentCartographer basic functionality."""

    def test_create_empty(self):
        """Test creating empty cartographer."""
        cart = ArgumentCartographer()
        assert cart.nodes == {}
        assert cart.edges == []
        assert cart.debate_id is None
        assert cart.topic is None

    def test_set_debate_context(self):
        """Test set_debate_context."""
        cart = ArgumentCartographer()
        cart.set_debate_context("debate-123", "Test topic")
        assert cart.debate_id == "debate-123"
        assert cart.topic == "Test topic"


# =============================================================================
# ArgumentCartographer Message Tests
# =============================================================================


class TestCartographerMessages:
    """Tests for ArgumentCartographer.update_from_message()."""

    def test_creates_node_from_message(self):
        """Test message creates a node."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message(
            agent="agent1",
            content="I propose we implement feature X.",
            role="proposer",
            round_num=1,
        )
        assert node_id in cart.nodes
        assert len(cart.nodes) == 1

    def test_node_has_correct_properties(self):
        """Test created node has correct properties."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message(
            agent="agent1",
            content="I propose we implement feature X.",
            role="proposer",
            round_num=1,
        )
        node = cart.nodes[node_id]
        assert node.agent == "agent1"
        assert node.node_type == NodeType.PROPOSAL
        assert node.round_num == 1
        assert "I propose" in node.summary

    def test_truncates_long_content_for_summary(self):
        """Test summary is truncated for long content."""
        cart = ArgumentCartographer()
        long_content = "A" * 200
        node_id = cart.update_from_message(
            agent="agent1",
            content=long_content,
            role="proposer",
            round_num=1,
        )
        node = cart.nodes[node_id]
        assert len(node.summary) <= 103  # 100 chars + "..."

    def test_stores_full_content(self):
        """Test full content is stored."""
        cart = ArgumentCartographer()
        content = "This is the full content of the message."
        node_id = cart.update_from_message(
            agent="agent1",
            content=content,
            role="proposer",
            round_num=1,
        )
        node = cart.nodes[node_id]
        assert node.full_content == content

    def test_stores_metadata(self):
        """Test metadata is stored."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message(
            agent="agent1",
            content="Content",
            role="proposer",
            round_num=1,
            metadata={"custom": "data"},
        )
        node = cart.nodes[node_id]
        assert node.metadata == {"custom": "data"}

    def test_returns_node_id(self):
        """Test returns node ID."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message(
            agent="agent1",
            content="Content",
            role="proposer",
            round_num=1,
        )
        assert isinstance(node_id, str)
        assert len(node_id) > 0

    def test_unique_node_ids(self):
        """Test node IDs are unique."""
        cart = ArgumentCartographer()
        id1 = cart.update_from_message("a1", "Content 1", "proposer", 1)
        id2 = cart.update_from_message("a2", "Content 2", "critic", 1)
        id3 = cart.update_from_message("a1", "Content 3", "proposer", 2)
        assert id1 != id2
        assert id2 != id3
        assert id1 != id3


# =============================================================================
# ArgumentCartographer Type Inference Tests
# =============================================================================


class TestCartographerTypeInference:
    """Tests for node type inference."""

    def test_infers_proposal_from_role(self):
        """Test infers PROPOSAL from proposer role."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("a", "Any content", "proposer", 1)
        assert cart.nodes[node_id].node_type == NodeType.PROPOSAL

    def test_infers_critique_from_role(self):
        """Test infers CRITIQUE from critic role."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("a", "Any content", "critic", 1)
        assert cart.nodes[node_id].node_type == NodeType.CRITIQUE

    def test_infers_proposal_from_content(self):
        """Test infers PROPOSAL from content signals."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("a", "I propose that we should use Python.", "", 1)
        assert cart.nodes[node_id].node_type == NodeType.PROPOSAL

    def test_infers_critique_from_content(self):
        """Test infers CRITIQUE from content signals."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("a", "I disagree with this approach.", "", 1)
        assert cart.nodes[node_id].node_type == NodeType.CRITIQUE

    def test_infers_concession_from_content(self):
        """Test infers CONCESSION from content signals."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("a", "I agree that's a good point.", "", 1)
        assert cart.nodes[node_id].node_type == NodeType.CONCESSION

    def test_infers_evidence_from_content(self):
        """Test infers EVIDENCE from content signals."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("a", "Research indicates this is true.", "", 1)
        assert cart.nodes[node_id].node_type == NodeType.EVIDENCE

    def test_defaults_to_evidence(self):
        """Test defaults to EVIDENCE for unknown content."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("a", "Some neutral statement.", "", 1)
        assert cart.nodes[node_id].node_type == NodeType.EVIDENCE


# =============================================================================
# ArgumentCartographer Critique Tests
# =============================================================================


class TestCartographerCritiques:
    """Tests for ArgumentCartographer.update_from_critique()."""

    def test_creates_edge_for_critique(self):
        """Test critique creates edge between agents."""
        cart = ArgumentCartographer()
        # Need nodes first
        cart.update_from_message("critic", "I disagree", "critic", 1)
        cart.update_from_message("target", "My proposal", "proposer", 1)

        edge_id = cart.update_from_critique("critic", "target", 0.8, 1)

        assert len(cart.edges) >= 1
        assert edge_id is not None

    def test_returns_none_without_nodes(self):
        """Test returns None if agents have no nodes."""
        cart = ArgumentCartographer()
        edge_id = cart.update_from_critique("critic", "target", 0.8, 1)
        assert edge_id is None

    def test_high_severity_creates_refutes(self):
        """Test high severity creates REFUTES relation."""
        cart = ArgumentCartographer()
        cart.update_from_message("critic", "Content", "", 1)
        cart.update_from_message("target", "Content", "", 1)

        cart.update_from_critique("critic", "target", 0.9, 1)

        # Find the critique edge
        critique_edges = [e for e in cart.edges if e.relation == EdgeRelation.REFUTES]
        assert len(critique_edges) >= 1

    def test_medium_severity_creates_modifies(self):
        """Test medium severity creates MODIFIES relation."""
        cart = ArgumentCartographer()
        cart.update_from_message("critic", "Content", "", 1)
        cart.update_from_message("target", "Content", "", 1)

        cart.update_from_critique("critic", "target", 0.5, 1)

        # Find the critique edge
        critique_edges = [e for e in cart.edges if e.relation == EdgeRelation.MODIFIES]
        assert len(critique_edges) >= 1

    def test_low_severity_creates_responds_to(self):
        """Test low severity creates RESPONDS_TO relation."""
        cart = ArgumentCartographer()
        cart.update_from_message("critic", "Content", "", 1)
        cart.update_from_message("target", "Content", "", 1)

        cart.update_from_critique("critic", "target", 0.2, 1)

        # Find the critique edge
        critique_edges = [e for e in cart.edges if e.relation == EdgeRelation.RESPONDS_TO]
        assert len(critique_edges) >= 1

    def test_stores_critique_text(self):
        """Test stores critique text in metadata."""
        cart = ArgumentCartographer()
        cart.update_from_message("critic", "Content", "", 1)
        cart.update_from_message("target", "Content", "", 1)

        cart.update_from_critique("critic", "target", 0.5, 1, "This is the critique text")

        # Find edges with critique_text metadata
        edges_with_text = [e for e in cart.edges if e.metadata.get("critique_text")]
        assert len(edges_with_text) >= 1


# =============================================================================
# ArgumentCartographer Vote Tests
# =============================================================================


class TestCartographerVotes:
    """Tests for ArgumentCartographer.update_from_vote()."""

    def test_creates_vote_node(self):
        """Test creates vote node."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_vote("agent1", "Option A", 1)

        assert node_id in cart.nodes
        node = cart.nodes[node_id]
        assert node.node_type == NodeType.VOTE

    def test_vote_summary_includes_value(self):
        """Test vote summary includes vote value."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_vote("agent1", "Yes", 1)

        node = cart.nodes[node_id]
        assert "Yes" in node.summary

    def test_vote_stores_value_in_metadata(self):
        """Test vote stores value in metadata."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_vote("agent1", "Option B", 1)

        node = cart.nodes[node_id]
        assert node.metadata["vote_value"] == "Option B"

    def test_vote_links_to_proposal(self):
        """Test vote links to round's proposal."""
        cart = ArgumentCartographer()
        # Create a proposal first
        cart.update_from_message("proposer", "I propose...", "proposer", 1)

        # Then vote
        cart.update_from_vote("voter", "Yes", 1)

        # Should have edge from vote to proposal
        vote_edges = [e for e in cart.edges if e.relation == EdgeRelation.RESPONDS_TO]
        assert len(vote_edges) >= 1


# =============================================================================
# ArgumentCartographer Consensus Tests
# =============================================================================


class TestCartographerConsensus:
    """Tests for ArgumentCartographer.update_from_consensus()."""

    def test_creates_consensus_node(self):
        """Test creates consensus node."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_consensus("Agreement reached", 3)

        assert node_id in cart.nodes
        node = cart.nodes[node_id]
        assert node.node_type == NodeType.CONSENSUS

    def test_consensus_node_id_format(self):
        """Test consensus node ID format."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_consensus("Result", 5)

        assert node_id == "consensus_5"

    def test_consensus_summary_includes_result(self):
        """Test consensus summary includes result."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_consensus("Unanimous yes", 1)

        node = cart.nodes[node_id]
        assert "Unanimous yes" in node.summary

    def test_consensus_stores_vote_counts(self):
        """Test consensus stores vote counts."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_consensus(
            "Majority",
            1,
            vote_counts={"yes": 3, "no": 1},
        )

        node = cart.nodes[node_id]
        assert node.metadata["vote_counts"] == {"yes": 3, "no": 1}

    def test_consensus_links_to_votes(self):
        """Test consensus links to votes in the round."""
        cart = ArgumentCartographer()
        # Create votes
        cart.update_from_vote("v1", "yes", 1)
        cart.update_from_vote("v2", "yes", 1)

        # Create consensus
        cart.update_from_consensus("Agreed", 1)

        # Should have edges from votes to consensus
        support_edges = [e for e in cart.edges if e.relation == EdgeRelation.SUPPORTS]
        assert len(support_edges) >= 2


# =============================================================================
# ArgumentCartographer Mermaid Export Tests
# =============================================================================


class TestCartographerMermaid:
    """Tests for ArgumentCartographer.export_mermaid()."""

    def test_exports_valid_mermaid(self):
        """Test exports valid Mermaid diagram."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Proposal", "proposer", 1)

        mermaid = cart.export_mermaid()

        assert mermaid.startswith("graph ")
        assert "classDef" in mermaid

    def test_direction_parameter(self):
        """Test respects direction parameter."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Content", "", 1)

        td = cart.export_mermaid(direction="TD")
        lr = cart.export_mermaid(direction="LR")

        assert "graph TD" in td
        assert "graph LR" in lr

    def test_includes_nodes(self):
        """Test includes nodes in output."""
        cart = ArgumentCartographer()
        cart.update_from_message("agent1", "My proposal", "proposer", 1)

        mermaid = cart.export_mermaid()

        assert "agent1" in mermaid

    def test_includes_round_subgraphs(self):
        """Test includes round subgraphs."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Content", "", 1)
        cart.update_from_message("a2", "Content", "", 2)

        mermaid = cart.export_mermaid()

        assert "Round_1" in mermaid
        assert "Round_2" in mermaid

    def test_includes_edges(self):
        """Test includes edges in output."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "I propose...", "proposer", 1)
        cart.update_from_message("a2", "I disagree...", "critic", 1)

        mermaid = cart.export_mermaid()

        # Should have arrow notation (may include labels like -.-|refutes|>)
        assert "-->" in mermaid or "-.-" in mermaid or "==>" in mermaid or "|>" in mermaid

    def test_sanitizes_special_characters(self):
        """Test sanitizes special characters."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", 'Content with "quotes" and [brackets]', "", 1)

        mermaid = cart.export_mermaid()

        # Should not have raw brackets or quotes that break Mermaid
        assert '"quotes"' not in mermaid
        assert "[brackets]" not in mermaid


# =============================================================================
# ArgumentCartographer JSON Export Tests
# =============================================================================


class TestCartographerJSON:
    """Tests for ArgumentCartographer.export_json()."""

    def test_exports_valid_json(self):
        """Test exports valid JSON."""
        cart = ArgumentCartographer()
        cart.set_debate_context("debate-1", "Test topic")
        cart.update_from_message("a1", "Content", "", 1)

        json_str = cart.export_json()
        data = json.loads(json_str)

        assert "debate_id" in data
        assert "nodes" in data
        assert "edges" in data

    def test_includes_debate_context(self):
        """Test includes debate context."""
        cart = ArgumentCartographer()
        cart.set_debate_context("debate-123", "Topic here")

        data = json.loads(cart.export_json())

        assert data["debate_id"] == "debate-123"
        assert data["topic"] == "Topic here"

    def test_includes_nodes(self):
        """Test includes nodes."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Content", "", 1)

        data = json.loads(cart.export_json())

        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["agent"] == "a1"

    def test_includes_edges(self):
        """Test includes edges."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "I propose...", "proposer", 1)
        cart.update_from_message("a2", "I disagree...", "critic", 1)

        data = json.loads(cart.export_json())

        assert len(data["edges"]) >= 1

    def test_includes_metadata(self):
        """Test includes metadata section."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Content", "", 1)

        data = json.loads(cart.export_json())

        assert "metadata" in data
        assert data["metadata"]["node_count"] == 1

    def test_include_full_content_parameter(self):
        """Test include_full_content parameter."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Full content here", "", 1)

        without = json.loads(cart.export_json(include_full_content=False))
        with_content = json.loads(cart.export_json(include_full_content=True))

        assert "full_content" not in without["nodes"][0]
        assert with_content["nodes"][0]["full_content"] == "Full content here"


# =============================================================================
# ArgumentCartographer Statistics Tests
# =============================================================================


class TestCartographerStatistics:
    """Tests for ArgumentCartographer.get_statistics()."""

    def test_returns_dict(self):
        """Test returns dictionary."""
        cart = ArgumentCartographer()
        stats = cart.get_statistics()
        assert isinstance(stats, dict)

    def test_empty_graph_stats(self):
        """Test stats for empty graph."""
        cart = ArgumentCartographer()
        stats = cart.get_statistics()

        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["max_depth"] == 0

    def test_node_count(self):
        """Test node count."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Content 1", "", 1)
        cart.update_from_message("a2", "Content 2", "", 1)
        cart.update_from_message("a3", "Content 3", "", 2)

        stats = cart.get_statistics()

        assert stats["node_count"] == 3

    def test_edge_count(self):
        """Test edge count."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "I propose...", "proposer", 1)
        cart.update_from_message("a2", "I disagree...", "critic", 1)

        stats = cart.get_statistics()

        assert stats["edge_count"] >= 1

    def test_node_types_breakdown(self):
        """Test node types breakdown."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "I propose...", "proposer", 1)
        cart.update_from_message("a2", "I disagree...", "critic", 1)

        stats = cart.get_statistics()

        assert "node_types" in stats
        assert stats["node_types"]["proposal"] >= 1
        assert stats["node_types"]["critique"] >= 1

    def test_agents_list(self):
        """Test agents list."""
        cart = ArgumentCartographer()
        cart.update_from_message("agent1", "Content", "", 1)
        cart.update_from_message("agent2", "Content", "", 1)

        stats = cart.get_statistics()

        assert "agent1" in stats["agents"]
        assert "agent2" in stats["agents"]

    def test_rounds_count(self):
        """Test rounds count."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Content", "", 1)
        cart.update_from_message("a1", "Content", "", 2)
        cart.update_from_message("a1", "Content", "", 3)

        stats = cart.get_statistics()

        assert stats["rounds"] == 3

    def test_complexity_score_range(self):
        """Test complexity score is in valid range."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "I propose...", "proposer", 1)
        cart.update_from_message("a2", "I disagree...", "critic", 1)
        cart.update_from_message("a1", "But consider...", "", 2)

        stats = cart.get_statistics()

        assert 0 <= stats["complexity_score"] <= 1

    def test_max_depth_calculation(self):
        """Test max depth calculation."""
        cart = ArgumentCartographer()
        # Create a chain
        cart.update_from_message("a1", "I propose...", "proposer", 1)
        cart.update_from_message("a2", "I disagree...", "critic", 1)

        stats = cart.get_statistics()

        assert stats["max_depth"] >= 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestCartographerEdgeCases:
    """Edge case tests for ArgumentCartographer."""

    def test_empty_content(self):
        """Test handles empty content."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("agent", "", "", 1)
        assert node_id in cart.nodes

    def test_very_long_content(self):
        """Test handles very long content."""
        cart = ArgumentCartographer()
        long_content = "X" * 10000
        node_id = cart.update_from_message("agent", long_content, "", 1)

        node = cart.nodes[node_id]
        assert len(node.summary) <= 103
        assert node.full_content == long_content

    def test_special_characters_in_agent_name(self):
        """Test handles special characters in agent name."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("agent-1_test", "Content", "", 1)
        assert node_id in cart.nodes

    def test_unicode_content(self):
        """Test handles unicode content."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message("agent", "日本語コンテンツ 🎯", "", 1)

        node = cart.nodes[node_id]
        assert "日本語" in node.summary

    def test_newlines_in_content(self):
        """Test handles newlines in content."""
        cart = ArgumentCartographer()
        content = "Line 1\nLine 2\nLine 3"
        node_id = cart.update_from_message("agent", content, "", 1)

        node = cart.nodes[node_id]
        # Summary should have newlines replaced
        assert "\n" not in node.summary

    def test_multiple_rounds(self):
        """Test handles multiple rounds."""
        cart = ArgumentCartographer()
        for i in range(10):
            cart.update_from_message(f"a{i % 3}", f"Content {i}", "", i)

        assert len(cart.nodes) == 10
        stats = cart.get_statistics()
        assert stats["rounds"] == 10

    def test_many_agents(self):
        """Test handles many agents."""
        cart = ArgumentCartographer()
        for i in range(20):
            cart.update_from_message(f"agent{i}", "Content", "", 1)

        stats = cart.get_statistics()
        assert len(stats["agents"]) == 20

    def test_cycle_detection_in_depth(self):
        """Test depth calculation handles potential cycles."""
        cart = ArgumentCartographer()
        # Create nodes that could form a cycle
        id1 = cart.update_from_message("a1", "Node 1", "", 1)
        id2 = cart.update_from_message("a2", "Node 2", "", 1)

        # Manually add a back-edge
        cart.edges.append(
            ArgumentEdge(
                source_id=id2,
                target_id=id1,
                relation=EdgeRelation.RESPONDS_TO,
            )
        )
        cart.edges.append(
            ArgumentEdge(
                source_id=id1,
                target_id=id2,
                relation=EdgeRelation.RESPONDS_TO,
            )
        )

        # Should not hang or crash
        stats = cart.get_statistics()
        assert "max_depth" in stats

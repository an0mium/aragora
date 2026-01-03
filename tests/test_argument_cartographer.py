"""Tests for the Argument Cartographer visualization system."""

import pytest
from aragora.visualization.mapper import (
    ArgumentCartographer,
    ArgumentNode,
    ArgumentEdge,
    NodeType,
    EdgeRelation,
)


class TestArgumentCartographer:
    """Test the core cartographer functionality."""

    def test_create_empty_cartographer(self):
        """Cartographer should initialize with empty graph."""
        cart = ArgumentCartographer()
        assert len(cart.nodes) == 0
        assert len(cart.edges) == 0

    def test_add_proposal_message(self):
        """Adding a proposal should create a node and track it."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message(
            agent="claude-visionary",
            content="I propose we implement feature X",
            role="proposer",
            round_num=1,
        )
        
        assert node_id in cart.nodes
        assert cart.nodes[node_id].node_type == NodeType.PROPOSAL
        assert cart.nodes[node_id].agent == "claude-visionary"
        assert cart.nodes[node_id].round_num == 1

    def test_add_critique_creates_edge(self):
        """A critique should link to the proposal."""
        cart = ArgumentCartographer()
        
        # Add proposal
        cart.update_from_message(
            agent="claude-visionary",
            content="I propose we implement feature X",
            role="proposer",
            round_num=1,
        )
        
        # Add critique
        cart.update_from_message(
            agent="codex-engineer",
            content="I disagree with this approach because...",
            role="critic",
            round_num=1,
        )
        
        assert len(cart.nodes) == 2
        assert len(cart.edges) >= 1
        
        # Find the critique->proposal edge
        refute_edges = [e for e in cart.edges if e.relation == EdgeRelation.REFUTES]
        assert len(refute_edges) >= 1

    def test_export_mermaid_format(self):
        """Mermaid export should produce valid syntax."""
        cart = ArgumentCartographer()
        cart.set_debate_context("test-123", "Test Topic")
        
        cart.update_from_message(
            agent="agent1",
            content="Proposal content",
            role="proposer",
            round_num=1,
        )
        
        mermaid = cart.export_mermaid()
        
        assert mermaid.startswith("graph TD")
        assert "classDef proposal" in mermaid
        assert "subgraph Round_1" in mermaid

    def test_export_json_structure(self):
        """JSON export should have correct structure."""
        cart = ArgumentCartographer()
        cart.set_debate_context("test-123", "Test Topic")
        
        cart.update_from_message(
            agent="agent1",
            content="Content",
            role="proposer",
            round_num=1,
        )
        
        import json
        data = json.loads(cart.export_json())
        
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert data["debate_id"] == "test-123"
        assert data["topic"] == "Test Topic"

    def test_statistics(self):
        """Statistics should accurately reflect graph state."""
        cart = ArgumentCartographer()
        
        cart.update_from_message("a1", "Proposal", "proposer", 1)
        cart.update_from_message("a2", "I disagree", "critic", 1)
        cart.update_from_message("a3", "Good point", "supporter", 1)
        
        stats = cart.get_statistics()
        
        assert stats["total_nodes"] == 3
        assert stats["rounds"] == 1
        assert "a1" in stats["agents"]
        assert "a2" in stats["agents"]

    def test_sanitize_special_characters(self):
        """Special characters should be sanitized for Mermaid."""
        cart = ArgumentCartographer()
        
        cart.update_from_message(
            agent="agent",
            content='Content with "quotes" and [brackets] and <angles>',
            role="proposer",
            round_num=1,
        )
        
        mermaid = cart.export_mermaid()
        
        # Should not contain raw problematic characters
        assert '"quotes"' not in mermaid or "'" in mermaid  # Quotes converted
        assert "[brackets]" not in mermaid  # Brackets converted

    def test_vote_links_to_proposal(self):
        """Votes should link to the round's proposal."""
        cart = ArgumentCartographer()
        
        cart.update_from_message("proposer", "My proposal", "proposer", 1)
        cart.update_from_vote("voter1", "approve", 1)
        
        assert len(cart.edges) >= 1
        vote_edges = [e for e in cart.edges if e.relation == EdgeRelation.RESPONDS_TO]
        assert len(vote_edges) >= 1

    def test_consensus_links_to_votes(self):
        """Consensus should link to all votes."""
        cart = ArgumentCartographer()
        
        cart.update_from_message("proposer", "Proposal", "proposer", 1)
        cart.update_from_vote("v1", "approve", 1)
        cart.update_from_vote("v2", "approve", 1)
        cart.update_from_consensus("approved", 1, {"approve": 2})
        
        consensus_nodes = [n for n in cart.nodes.values() if n.node_type == NodeType.CONSENSUS]
        assert len(consensus_nodes) == 1
        
        # Votes should link to consensus
        support_edges = [e for e in cart.edges if e.relation == EdgeRelation.SUPPORTS]
        assert len(support_edges) >= 2


class TestNodeTypeInference:
    """Test the content-based node type inference."""

    def test_infer_proposal_from_content(self):
        """'I propose' should trigger PROPOSAL type."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message(
            agent="agent",
            content="I propose we should refactor this module",
            role="",  # No role hint
            round_num=1,
        )
        assert cart.nodes[node_id].node_type == NodeType.PROPOSAL

    def test_infer_concession_from_content(self):
        """'Good point' should trigger CONCESSION type."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message(
            agent="agent",
            content="Good point, I agree with your assessment",
            role="",
            round_num=1,
        )
        assert cart.nodes[node_id].node_type == NodeType.CONCESSION

    def test_infer_critique_from_content(self):
        """'I disagree' should trigger CRITIQUE type."""
        cart = ArgumentCartographer()
        node_id = cart.update_from_message(
            agent="agent",
            content="However, I disagree with this approach",
            role="",
            round_num=1,
        )
        assert cart.nodes[node_id].node_type == NodeType.CRITIQUE
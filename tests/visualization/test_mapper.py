"""Tests for argument cartographer and graph visualization."""

import json
import pytest

from aragora.visualization.mapper import (
    ArgumentCartographer,
    ArgumentEdge,
    ArgumentNode,
    EdgeRelation,
    NodeType,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_node_types_exist(self):
        """Should have all expected node types."""
        expected = [
            "proposal",
            "critique",
            "evidence",
            "concession",
            "rebuttal",
            "vote",
            "consensus",
        ]
        actual = [t.value for t in NodeType]
        assert sorted(actual) == sorted(expected)

    def test_node_type_values(self):
        """Node type values should match enum names."""
        assert NodeType.PROPOSAL.value == "proposal"
        assert NodeType.CRITIQUE.value == "critique"
        assert NodeType.CONSENSUS.value == "consensus"


class TestEdgeRelation:
    """Tests for EdgeRelation enum."""

    def test_all_edge_relations_exist(self):
        """Should have all expected edge relations."""
        expected = ["supports", "refutes", "modifies", "responds_to", "concedes_to"]
        actual = [r.value for r in EdgeRelation]
        assert sorted(actual) == sorted(expected)

    def test_edge_relation_values(self):
        """Edge relation values should match enum names."""
        assert EdgeRelation.SUPPORTS.value == "supports"
        assert EdgeRelation.REFUTES.value == "refutes"
        assert EdgeRelation.CONCEDES_TO.value == "concedes_to"


class TestArgumentNode:
    """Tests for ArgumentNode dataclass."""

    def test_create_node(self):
        """Basic node creation."""
        node = ArgumentNode(
            id="node-1",
            agent="claude",
            node_type=NodeType.PROPOSAL,
            summary="Test proposal",
            round_num=1,
            timestamp=1234567890.0,
        )
        assert node.id == "node-1"
        assert node.agent == "claude"
        assert node.node_type == NodeType.PROPOSAL
        assert node.summary == "Test proposal"
        assert node.round_num == 1

    def test_node_defaults(self):
        """Node should have default values."""
        node = ArgumentNode(
            id="n1",
            agent="a",
            node_type=NodeType.EVIDENCE,
            summary="test",
            round_num=1,
            timestamp=0.0,
        )
        assert node.full_content is None
        assert node.metadata == {}

    def test_node_to_dict(self):
        """Should serialize to dictionary."""
        node = ArgumentNode(
            id="n1",
            agent="claude",
            node_type=NodeType.CRITIQUE,
            summary="Critical point",
            round_num=2,
            timestamp=1000.0,
            metadata={"key": "value"},
        )
        data = node.to_dict()

        assert data["id"] == "n1"
        assert data["agent"] == "claude"
        assert data["node_type"] == "critique"
        assert data["summary"] == "Critical point"
        assert data["round_num"] == 2
        assert data["timestamp"] == 1000.0
        assert data["metadata"] == {"key": "value"}


class TestArgumentEdge:
    """Tests for ArgumentEdge dataclass."""

    def test_create_edge(self):
        """Basic edge creation."""
        edge = ArgumentEdge(
            source_id="n1",
            target_id="n2",
            relation=EdgeRelation.SUPPORTS,
        )
        assert edge.source_id == "n1"
        assert edge.target_id == "n2"
        assert edge.relation == EdgeRelation.SUPPORTS

    def test_edge_defaults(self):
        """Edge should have default values."""
        edge = ArgumentEdge(
            source_id="n1",
            target_id="n2",
            relation=EdgeRelation.REFUTES,
        )
        assert edge.weight == 1.0
        assert edge.metadata == {}

    def test_edge_to_dict(self):
        """Should serialize to dictionary."""
        edge = ArgumentEdge(
            source_id="n1",
            target_id="n2",
            relation=EdgeRelation.MODIFIES,
            weight=0.7,
            metadata={"reason": "test"},
        )
        data = edge.to_dict()

        assert data["source_id"] == "n1"
        assert data["target_id"] == "n2"
        assert data["relation"] == "modifies"
        assert data["weight"] == 0.7
        assert data["metadata"] == {"reason": "test"}


class TestArgumentCartographer:
    """Tests for ArgumentCartographer."""

    @pytest.fixture
    def cartographer(self):
        """Create a fresh cartographer."""
        return ArgumentCartographer()

    def test_init(self, cartographer):
        """Should initialize with empty collections."""
        assert cartographer.nodes == {}
        assert cartographer.edges == []
        assert cartographer.debate_id is None
        assert cartographer.topic is None

    def test_set_debate_context(self, cartographer):
        """Should set debate context."""
        cartographer.set_debate_context("debate-123", "Test topic")
        assert cartographer.debate_id == "debate-123"
        assert cartographer.topic == "Test topic"

    def test_update_from_message_creates_node(self, cartographer):
        """Should create a node from a message."""
        node_id = cartographer.update_from_message(
            agent="claude",
            content="I propose we implement feature X",
            role="proposer",
            round_num=1,
        )

        assert node_id in cartographer.nodes
        node = cartographer.nodes[node_id]
        assert node.agent == "claude"
        assert node.round_num == 1
        assert node.node_type == NodeType.PROPOSAL

    def test_update_from_message_truncates_summary(self, cartographer):
        """Long content should be truncated in summary."""
        long_content = "A" * 150
        node_id = cartographer.update_from_message(
            agent="test",
            content=long_content,
            role="agent",
            round_num=1,
        )

        node = cartographer.nodes[node_id]
        assert len(node.summary) <= 103  # 100 + "..."
        assert node.full_content == long_content

    def test_update_from_message_infers_proposal_type(self, cartographer):
        """Should infer proposal type from content."""
        node_id = cartographer.update_from_message(
            agent="agent",
            content="I propose that we should implement this feature",
            role="agent",
            round_num=1,
        )
        assert cartographer.nodes[node_id].node_type == NodeType.PROPOSAL

    def test_update_from_message_infers_critique_type(self, cartographer):
        """Should infer critique type from content."""
        # First add a proposal
        cartographer.update_from_message(
            agent="proposer",
            content="My proposal",
            role="proposer",
            round_num=1,
        )

        node_id = cartographer.update_from_message(
            agent="critic",
            content="I disagree with this approach, there's a major issue",
            role="agent",
            round_num=1,
        )
        assert cartographer.nodes[node_id].node_type == NodeType.CRITIQUE

    def test_update_from_message_infers_concession_type(self, cartographer):
        """Should infer concession type from content."""
        node_id = cartographer.update_from_message(
            agent="agent",
            content="I agree, that's a good point you made",
            role="agent",
            round_num=1,
        )
        assert cartographer.nodes[node_id].node_type == NodeType.CONCESSION

    def test_update_from_vote(self, cartographer):
        """Should create vote node."""
        # Add proposal first
        cartographer.update_from_message(
            agent="proposer",
            content="My proposal",
            role="proposer",
            round_num=1,
        )

        node_id = cartographer.update_from_vote("claude", "approve", 1)

        assert node_id in cartographer.nodes
        node = cartographer.nodes[node_id]
        assert node.node_type == NodeType.VOTE
        assert node.agent == "claude"
        assert "approve" in node.summary

    def test_update_from_consensus(self, cartographer):
        """Should create consensus node."""
        node_id = cartographer.update_from_consensus(
            result="approved",
            round_num=3,
            vote_counts={"approve": 3, "reject": 1},
        )

        assert node_id in cartographer.nodes
        node = cartographer.nodes[node_id]
        assert node.node_type == NodeType.CONSENSUS
        assert node.agent == "system"
        assert node.metadata["result"] == "approved"

    def test_update_from_critique(self, cartographer):
        """Should create critique edge between agents."""
        # Add messages from both agents
        cartographer.update_from_message(
            agent="target",
            content="Original proposal",
            role="proposer",
            round_num=1,
        )
        cartographer.update_from_message(
            agent="critic",
            content="My critique",
            role="critic",
            round_num=1,
        )

        edge_id = cartographer.update_from_critique(
            critic_agent="critic",
            target_agent="target",
            severity=0.8,
            round_num=1,
        )

        assert edge_id is not None
        assert len(cartographer.edges) >= 1

    def test_export_json(self, cartographer):
        """Should export to valid JSON."""
        cartographer.set_debate_context("test-debate", "Test topic")
        cartographer.update_from_message(
            agent="claude",
            content="Test proposal",
            role="proposer",
            round_num=1,
        )

        json_str = cartographer.export_json()
        data = json.loads(json_str)

        assert data["debate_id"] == "test-debate"
        assert data["topic"] == "Test topic"
        assert len(data["nodes"]) == 1
        assert "metadata" in data

    def test_export_json_includes_full_content(self, cartographer):
        """Should optionally include full content."""
        cartographer.update_from_message(
            agent="agent",
            content="Full content here",
            role="agent",
            round_num=1,
        )

        json_str = cartographer.export_json(include_full_content=True)
        data = json.loads(json_str)

        assert data["nodes"][0]["full_content"] == "Full content here"

    def test_export_mermaid(self, cartographer):
        """Should export to Mermaid format."""
        cartographer.update_from_message(
            agent="claude",
            content="Test proposal",
            role="proposer",
            round_num=1,
        )

        mermaid = cartographer.export_mermaid()

        assert "graph" in mermaid
        assert "Round" in mermaid
        assert "classDef" in mermaid

    def test_export_mermaid_with_direction(self, cartographer):
        """Should respect direction parameter."""
        cartographer.update_from_message(
            agent="agent",
            content="Test",
            role="agent",
            round_num=1,
        )

        mermaid_td = cartographer.export_mermaid(direction="TD")
        mermaid_lr = cartographer.export_mermaid(direction="LR")

        assert "graph TD" in mermaid_td
        assert "graph LR" in mermaid_lr

    def test_get_statistics(self, cartographer):
        """Should return statistics."""
        cartographer.update_from_message(
            agent="claude",
            content="Proposal",
            role="proposer",
            round_num=1,
        )
        cartographer.update_from_message(
            agent="gemini",
            content="I disagree with this",
            role="critic",
            round_num=1,
        )

        stats = cartographer.get_statistics()

        assert stats["node_count"] == 2
        assert "edge_count" in stats
        assert "max_depth" in stats
        assert "complexity_score" in stats
        assert "agents" in stats
        assert len(stats["agents"]) == 2


class TestCartographerEdges:
    """Tests for edge creation in cartographer."""

    @pytest.fixture
    def populated_cartographer(self):
        """Create cartographer with some nodes."""
        cart = ArgumentCartographer()
        cart.set_debate_context("test", "Test debate")

        # Add proposal
        cart.update_from_message(
            agent="proposer",
            content="I propose we implement feature X",
            role="proposer",
            round_num=1,
        )

        # Add critique
        cart.update_from_message(
            agent="critic",
            content="I disagree, there's an issue with this approach",
            role="critic",
            round_num=1,
        )

        return cart

    def test_critique_links_to_proposal(self, populated_cartographer):
        """Critique should link to the round's proposal."""
        edges = populated_cartographer.edges

        # Should have at least one edge
        assert len(edges) >= 1

        # Find refutes edge
        refutes_edges = [e for e in edges if e.relation == EdgeRelation.REFUTES]
        assert len(refutes_edges) >= 1

    def test_vote_links_to_proposal(self):
        """Vote should link to the round's proposal."""
        cart = ArgumentCartographer()

        # Add proposal
        cart.update_from_message(
            agent="proposer",
            content="My proposal",
            role="proposer",
            round_num=1,
        )

        # Add vote
        cart.update_from_vote("voter", "approve", 1)

        # Should have edge from vote to proposal
        responds_edges = [e for e in cart.edges if e.relation == EdgeRelation.RESPONDS_TO]
        assert len(responds_edges) >= 1

    def test_consensus_links_to_votes(self):
        """Consensus should link to all votes in the round."""
        cart = ArgumentCartographer()

        # Add proposal
        cart.update_from_message(
            agent="proposer",
            content="My proposal",
            role="proposer",
            round_num=1,
        )

        # Add votes
        cart.update_from_vote("voter1", "approve", 1)
        cart.update_from_vote("voter2", "approve", 1)

        # Add consensus
        cart.update_from_consensus("approved", 1)

        # Should have edges from votes to consensus
        supports_edges = [e for e in cart.edges if e.relation == EdgeRelation.SUPPORTS]
        assert len(supports_edges) >= 2


class TestCartographerSanitization:
    """Tests for Mermaid sanitization."""

    @pytest.fixture
    def cartographer(self):
        """Create a fresh cartographer."""
        return ArgumentCartographer()

    def test_sanitizes_quotes(self, cartographer):
        """Should sanitize quotes for Mermaid."""
        cartographer.update_from_message(
            agent="agent",
            content='Test "quoted" content',
            role="agent",
            round_num=1,
        )

        mermaid = cartographer.export_mermaid()

        # Should not have double quotes inside node labels
        # (except the enclosing ones)
        assert '\\"' not in mermaid or "'" in mermaid

    def test_sanitizes_special_characters(self, cartographer):
        """Should sanitize special characters for Mermaid."""
        cartographer.update_from_message(
            agent="agent",
            content="Test with [brackets] and {braces} and <angles>",
            role="agent",
            round_num=1,
        )

        mermaid = cartographer.export_mermaid()

        # Check that special chars are replaced
        # (actual node content is in the summary which gets sanitized)
        assert "[brackets]" not in mermaid

    def test_sanitizes_newlines(self, cartographer):
        """Should sanitize newlines for Mermaid."""
        cartographer.update_from_message(
            agent="agent",
            content="Line 1\nLine 2\nLine 3",
            role="agent",
            round_num=1,
        )

        mermaid = cartographer.export_mermaid()

        # Newlines should be replaced with spaces in summaries
        assert "Line 1 Line 2" in mermaid or "Line 1  Line 2" in mermaid


class TestCartographerStatistics:
    """Tests for statistics calculation."""

    def test_empty_cartographer_stats(self):
        """Empty cartographer should return zero stats."""
        cart = ArgumentCartographer()
        stats = cart.get_statistics()

        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["max_depth"] == 0

    def test_complexity_score_bounds(self):
        """Complexity score should be between 0 and 1."""
        cart = ArgumentCartographer()

        # Add many nodes to increase complexity
        for i in range(10):
            cart.update_from_message(
                agent=f"agent{i % 3}",
                content=f"Message {i}",
                role="agent",
                round_num=i // 3 + 1,
            )

        stats = cart.get_statistics()

        assert 0 <= stats["complexity_score"] <= 1

    def test_max_depth_calculation(self):
        """Should calculate max depth correctly."""
        cart = ArgumentCartographer()

        # Create chain of nodes across rounds
        cart.update_from_message(
            agent="agent1",
            content="I propose this",
            role="proposer",
            round_num=1,
        )
        cart.update_from_message(
            agent="agent2",
            content="I disagree",
            role="critic",
            round_num=1,
        )
        cart.update_from_message(
            agent="agent1",
            content="I respond with evidence",
            role="proposer",
            round_num=2,
        )

        stats = cart.get_statistics()

        assert stats["max_depth"] >= 1

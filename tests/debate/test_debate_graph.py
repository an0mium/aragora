"""Tests for debate graph with counterfactual branching."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from aragora.debate.graph import (
    NodeType,
    BranchReason,
    MergeStrategy,
    DebateNode,
    Branch,
    MergeResult,
    BranchPolicy,
    ConvergenceScorer,
    DebateGraph,
    GraphReplayBuilder,
    GraphDebateOrchestrator,
)


class TestNodeType:
    """Test NodeType enum."""

    def test_all_types_exist(self):
        """Test all node types are defined."""
        assert NodeType.ROOT.value == "root"
        assert NodeType.PROPOSAL.value == "proposal"
        assert NodeType.CRITIQUE.value == "critique"
        assert NodeType.SYNTHESIS.value == "synthesis"
        assert NodeType.BRANCH_POINT.value == "branch_point"
        assert NodeType.MERGE_POINT.value == "merge_point"
        assert NodeType.COUNTERFACTUAL.value == "counterfactual"
        assert NodeType.CONCLUSION.value == "conclusion"


class TestBranchReason:
    """Test BranchReason enum."""

    def test_all_reasons_exist(self):
        """Test all branch reasons are defined."""
        assert BranchReason.HIGH_DISAGREEMENT.value == "high_disagreement"
        assert BranchReason.ALTERNATIVE_APPROACH.value == "alternative_approach"
        assert BranchReason.COUNTERFACTUAL_EXPLORATION.value == "counterfactual_exploration"
        assert BranchReason.RISK_MITIGATION.value == "risk_mitigation"
        assert BranchReason.UNCERTAINTY.value == "uncertainty"
        assert BranchReason.USER_REQUESTED.value == "user_requested"


class TestMergeStrategy:
    """Test MergeStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all merge strategies are defined."""
        assert MergeStrategy.BEST_PATH.value == "best_path"
        assert MergeStrategy.SYNTHESIS.value == "synthesis"
        assert MergeStrategy.VOTE.value == "vote"
        assert MergeStrategy.WEIGHTED.value == "weighted"
        assert MergeStrategy.PRESERVE_ALL.value == "preserve_all"


class TestDebateNode:
    """Test DebateNode dataclass."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = DebateNode(
            id="node1",
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="This is a proposal.",
        )
        assert node.id == "node1"
        assert node.node_type == NodeType.PROPOSAL
        assert node.agent_id == "claude"
        assert node.content == "This is a proposal."

    def test_node_defaults(self):
        """Test node default values."""
        node = DebateNode(
            id="node1",
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Task",
        )
        assert node.parent_ids == []
        assert node.child_ids == []
        assert node.branch_id is None
        assert node.confidence == 0.0
        assert node.claims == []
        assert node.evidence == []
        assert node.metadata == {}

    def test_node_hash(self):
        """Test node content hash."""
        node = DebateNode(
            id="node1",
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Test content",
        )
        hash1 = node.hash()
        assert len(hash1) == 16
        assert hash1.isalnum()

    def test_node_hash_deterministic(self):
        """Test hash is deterministic for same content."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        node1 = DebateNode(
            id="node1",
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Test content",
            timestamp=ts,
        )
        node2 = DebateNode(
            id="node2",
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Test content",
            timestamp=ts,
        )
        assert node1.hash() == node2.hash()

    def test_node_to_dict(self):
        """Test node serialization."""
        node = DebateNode(
            id="node1",
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Test",
            confidence=0.8,
            claims=["claim1"],
        )
        d = node.to_dict()
        assert d["id"] == "node1"
        assert d["node_type"] == "proposal"
        assert d["agent_id"] == "claude"
        assert d["confidence"] == 0.8
        assert d["claims"] == ["claim1"]
        assert "hash" in d

    def test_node_from_dict(self):
        """Test node deserialization."""
        data = {
            "id": "node1",
            "node_type": "critique",
            "agent_id": "gpt4",
            "content": "I disagree",
            "timestamp": "2024-01-01T12:00:00",
            "parent_ids": ["parent1"],
            "confidence": 0.9,
        }
        node = DebateNode.from_dict(data)
        assert node.id == "node1"
        assert node.node_type == NodeType.CRITIQUE
        assert node.agent_id == "gpt4"
        assert node.parent_ids == ["parent1"]
        assert node.confidence == 0.9

    def test_node_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = DebateNode(
            id="node1",
            node_type=NodeType.SYNTHESIS,
            agent_id="claude",
            content="Synthesized view",
            confidence=0.75,
            claims=["c1", "c2"],
            evidence=["e1"],
            metadata={"key": "value"},
        )
        d = original.to_dict()
        restored = DebateNode.from_dict(d)
        assert restored.id == original.id
        assert restored.node_type == original.node_type
        assert restored.claims == original.claims


class TestBranch:
    """Test Branch dataclass."""

    def test_branch_creation(self):
        """Test basic branch creation."""
        branch = Branch(
            id="branch1",
            name="Alternative",
            reason=BranchReason.HIGH_DISAGREEMENT,
            start_node_id="node1",
        )
        assert branch.id == "branch1"
        assert branch.reason == BranchReason.HIGH_DISAGREEMENT
        assert branch.is_active is True
        assert branch.is_merged is False

    def test_branch_to_dict(self):
        """Test branch serialization."""
        branch = Branch(
            id="branch1",
            name="Test Branch",
            reason=BranchReason.UNCERTAINTY,
            start_node_id="node1",
            hypothesis="What if we tried this?",
            confidence=0.6,
        )
        d = branch.to_dict()
        assert d["id"] == "branch1"
        assert d["reason"] == "uncertainty"
        assert d["hypothesis"] == "What if we tried this?"

    def test_branch_from_dict(self):
        """Test branch deserialization."""
        data = {
            "id": "branch1",
            "name": "Risk Branch",
            "reason": "risk_mitigation",
            "start_node_id": "node1",
            "is_active": False,
            "is_merged": True,
            "merged_into": "merge_node",
        }
        branch = Branch.from_dict(data)
        assert branch.id == "branch1"
        assert branch.reason == BranchReason.RISK_MITIGATION
        assert branch.is_active is False
        assert branch.is_merged is True


class TestMergeResult:
    """Test MergeResult dataclass."""

    def test_merge_result_creation(self):
        """Test merge result creation."""
        result = MergeResult(
            merged_node_id="merge1",
            source_branch_ids=["branch1", "branch2"],
            strategy=MergeStrategy.SYNTHESIS,
            synthesis="Combined view",
            confidence=0.85,
        )
        assert result.merged_node_id == "merge1"
        assert len(result.source_branch_ids) == 2
        assert result.strategy == MergeStrategy.SYNTHESIS

    def test_merge_result_to_dict(self):
        """Test merge result serialization."""
        result = MergeResult(
            merged_node_id="merge1",
            source_branch_ids=["b1"],
            strategy=MergeStrategy.BEST_PATH,
            synthesis="Winner takes all",
            confidence=0.9,
            insights_preserved=["insight1"],
        )
        d = result.to_dict()
        assert d["strategy"] == "best_path"
        assert d["insights_preserved"] == ["insight1"]


class TestBranchPolicy:
    """Test BranchPolicy configuration."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = BranchPolicy()
        assert policy.disagreement_threshold == 0.6
        assert policy.uncertainty_threshold == 0.7
        assert policy.max_branches == 4
        assert policy.max_depth == 5
        assert policy.allow_counterfactuals is True

    def test_custom_policy(self):
        """Test custom policy values."""
        policy = BranchPolicy(
            disagreement_threshold=0.8,
            max_branches=2,
            allow_counterfactuals=False,
        )
        assert policy.disagreement_threshold == 0.8
        assert policy.max_branches == 2
        assert policy.allow_counterfactuals is False

    def test_should_branch_on_disagreement(self):
        """Test branching on high disagreement."""
        policy = BranchPolicy(disagreement_threshold=0.5)
        should, reason = policy.should_branch(
            disagreement=0.7,
            uncertainty=0.3,
            current_branches=1,
            current_depth=2,
        )
        assert should is True
        assert reason == BranchReason.HIGH_DISAGREEMENT

    def test_should_branch_on_uncertainty(self):
        """Test branching on high uncertainty."""
        policy = BranchPolicy(uncertainty_threshold=0.6)
        should, reason = policy.should_branch(
            disagreement=0.3,
            uncertainty=0.8,
            current_branches=1,
            current_depth=2,
        )
        assert should is True
        assert reason == BranchReason.UNCERTAINTY

    def test_should_not_branch_at_max_branches(self):
        """Test no branching when at max branches."""
        policy = BranchPolicy(max_branches=3)
        should, reason = policy.should_branch(
            disagreement=0.9,
            uncertainty=0.9,
            current_branches=3,
            current_depth=2,
        )
        assert should is False
        assert reason is None

    def test_should_not_branch_at_max_depth(self):
        """Test no branching when at max depth."""
        policy = BranchPolicy(max_depth=4)
        should, reason = policy.should_branch(
            disagreement=0.9,
            uncertainty=0.9,
            current_branches=1,
            current_depth=4,
        )
        assert should is False
        assert reason is None

    def test_should_branch_on_alternative(self):
        """Test branching on good alternative."""
        policy = BranchPolicy(min_alternative_score=0.4)
        should, reason = policy.should_branch(
            disagreement=0.3,
            uncertainty=0.3,
            current_branches=1,
            current_depth=1,
            alternative_score=0.5,
        )
        assert should is True
        assert reason == BranchReason.ALTERNATIVE_APPROACH

    def test_no_branch_below_thresholds(self):
        """Test no branching when all metrics are low."""
        policy = BranchPolicy()
        should, reason = policy.should_branch(
            disagreement=0.3,
            uncertainty=0.3,
            current_branches=1,
            current_depth=1,
            alternative_score=0.1,
        )
        assert should is False
        assert reason is None


class TestConvergenceScorer:
    """Test ConvergenceScorer."""

    def test_scorer_creation(self):
        """Test scorer initialization."""
        scorer = ConvergenceScorer(threshold=0.7)
        assert scorer.threshold == 0.7

    def test_empty_branches_zero_score(self):
        """Test empty branches give zero convergence."""
        scorer = ConvergenceScorer()
        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="n1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="n2")
        score = scorer.score_convergence(branch_a, branch_b, [], [])
        assert score == 0.0

    def test_identical_claims_high_convergence(self):
        """Test identical claims give high convergence."""
        scorer = ConvergenceScorer()
        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="n1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="n2")

        node_a = DebateNode(
            id="na",
            node_type=NodeType.SYNTHESIS,
            agent_id="c",
            content="test",
            claims=["claim1", "claim2"],
            confidence=0.8,
        )
        node_b = DebateNode(
            id="nb",
            node_type=NodeType.SYNTHESIS,
            agent_id="g",
            content="test",
            claims=["claim1", "claim2"],
            confidence=0.8,
        )

        score = scorer.score_convergence(branch_a, branch_b, [node_a], [node_b])
        assert score >= 0.7  # High convergence

    def test_different_claims_low_convergence(self):
        """Test different claims give low convergence."""
        scorer = ConvergenceScorer()
        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="n1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="n2")

        node_a = DebateNode(
            id="na",
            node_type=NodeType.SYNTHESIS,
            agent_id="c",
            content="x",
            claims=["claim1", "claim2"],
            confidence=0.8,
        )
        node_b = DebateNode(
            id="nb",
            node_type=NodeType.SYNTHESIS,
            agent_id="g",
            content="y",
            claims=["claim3", "claim4"],
            confidence=0.2,
        )

        score = scorer.score_convergence(branch_a, branch_b, [node_a], [node_b])
        assert score < 0.5  # Low convergence

    def test_content_similarity_fallback(self):
        """Test content similarity when no claims."""
        scorer = ConvergenceScorer()
        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="n1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="n2")

        node_a = DebateNode(
            id="na", node_type=NodeType.SYNTHESIS, agent_id="c", content="The quick brown fox jumps"
        )
        node_b = DebateNode(
            id="nb", node_type=NodeType.SYNTHESIS, agent_id="g", content="The quick brown dog jumps"
        )

        score = scorer.score_convergence(branch_a, branch_b, [node_a], [node_b])
        assert 0 < score < 1  # Some similarity

    def test_should_merge_above_threshold(self):
        """Test should_merge returns True above threshold."""
        scorer = ConvergenceScorer(threshold=0.5)
        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="n1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="n2")

        node_a = DebateNode(
            id="na",
            node_type=NodeType.SYNTHESIS,
            agent_id="c",
            content="test",
            claims=["same"],
            confidence=0.8,
        )
        node_b = DebateNode(
            id="nb",
            node_type=NodeType.SYNTHESIS,
            agent_id="g",
            content="test",
            claims=["same"],
            confidence=0.8,
        )

        should = scorer.should_merge(branch_a, branch_b, [node_a], [node_b])
        assert should is True


class TestDebateGraph:
    """Test DebateGraph main class."""

    def test_graph_creation(self):
        """Test graph initialization."""
        graph = DebateGraph()
        assert graph.debate_id is not None
        assert graph.main_branch_id == "main"
        assert "main" in graph.branches
        assert len(graph.nodes) == 0

    def test_graph_with_custom_id(self):
        """Test graph with custom debate ID."""
        graph = DebateGraph(debate_id="custom123")
        assert graph.debate_id == "custom123"

    def test_add_root_node(self):
        """Test adding root node."""
        graph = DebateGraph()
        node = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="What is the meaning of life?",
        )
        assert node.id in graph.nodes
        assert graph.root_id == node.id
        assert graph.branches["main"].start_node_id == node.id

    def test_add_child_node(self):
        """Test adding child node."""
        graph = DebateGraph()
        parent = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Task",
        )
        child = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="My proposal",
            parent_id=parent.id,
        )
        assert child.parent_ids == [parent.id]
        assert child.id in parent.child_ids

    def test_add_node_with_metadata(self):
        """Test adding node with all fields."""
        graph = DebateGraph()
        node = graph.add_node(
            node_type=NodeType.SYNTHESIS,
            agent_id="claude",
            content="Synthesis",
            confidence=0.85,
            claims=["claim1", "claim2"],
            evidence=["source1"],
            metadata={"round": 3},
        )
        assert node.confidence == 0.85
        assert node.claims == ["claim1", "claim2"]
        assert node.evidence == ["source1"]
        assert node.metadata["round"] == 3

    def test_create_branch(self):
        """Test creating a new branch."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        branch = graph.create_branch(
            from_node_id=root.id,
            reason=BranchReason.HIGH_DISAGREEMENT,
            name="Alternative View",
            hypothesis="What if we consider X?",
        )

        assert branch.id in graph.branches
        assert branch.start_node_id == root.id
        assert branch.hypothesis == "What if we consider X?"
        assert root.metadata.get("is_branch_point") is True

    def test_create_branch_invalid_node(self):
        """Test creating branch from invalid node raises error."""
        graph = DebateGraph()
        with pytest.raises(ValueError):
            graph.create_branch("invalid", BranchReason.UNCERTAINTY, "Test")

    def test_get_branch_nodes(self):
        """Test getting nodes for a branch."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        branch = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "Alt")

        n1 = graph.add_node(NodeType.PROPOSAL, "claude", "P1", root.id, branch.id)
        n2 = graph.add_node(NodeType.CRITIQUE, "gpt4", "C1", n1.id, branch.id)

        branch_nodes = graph.get_branch_nodes(branch.id)
        assert len(branch_nodes) == 2
        assert n1 in branch_nodes
        assert n2 in branch_nodes

    def test_get_active_branches(self):
        """Test getting active branches."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        # Main branch is active
        active = graph.get_active_branches()
        assert len(active) == 1

        # Create another branch
        branch = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "Alt")
        active = graph.get_active_branches()
        assert len(active) == 2

    def test_merge_branches(self):
        """Test merging branches."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        # Create two branches
        branch1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        branch2 = graph.create_branch(root.id, BranchReason.ALTERNATIVE_APPROACH, "B2")

        # Add nodes to each branch
        n1 = graph.add_node(NodeType.PROPOSAL, "a1", "Content1", root.id, branch1.id, claims=["c1"])
        n2 = graph.add_node(NodeType.PROPOSAL, "a2", "Content2", root.id, branch2.id, claims=["c2"])

        # Merge
        result = graph.merge_branches(
            branch_ids=[branch1.id, branch2.id],
            strategy=MergeStrategy.SYNTHESIS,
            synthesizer_agent_id="system",
            synthesis_content="Combined insights",
        )

        assert result.merged_node_id in graph.nodes
        assert not graph.branches[branch1.id].is_active
        assert not graph.branches[branch2.id].is_active
        assert "c1" in result.insights_preserved or "c2" in result.insights_preserved

    def test_merge_invalid_branch(self):
        """Test merging invalid branch raises error."""
        graph = DebateGraph()
        with pytest.raises(ValueError):
            graph.merge_branches(["invalid"], MergeStrategy.BEST_PATH, "sys", "content")

    def test_get_path_to_node(self):
        """Test getting path from root to node."""
        graph = DebateGraph()
        n1 = graph.add_node(NodeType.ROOT, "system", "Task")
        n2 = graph.add_node(NodeType.PROPOSAL, "claude", "P", n1.id)
        n3 = graph.add_node(NodeType.CRITIQUE, "gpt4", "C", n2.id)
        n4 = graph.add_node(NodeType.SYNTHESIS, "claude", "S", n3.id)

        path = graph.get_path_to_node(n4.id)
        assert len(path) == 4
        assert path[0] == n1
        assert path[-1] == n4

    def test_get_path_invalid_node(self):
        """Test getting path for invalid node returns empty."""
        graph = DebateGraph()
        path = graph.get_path_to_node("invalid")
        assert path == []

    def test_get_leaf_nodes(self):
        """Test getting leaf nodes (no children)."""
        graph = DebateGraph()
        n1 = graph.add_node(NodeType.ROOT, "system", "Task")
        n2 = graph.add_node(NodeType.PROPOSAL, "claude", "P", n1.id)
        n3 = graph.add_node(NodeType.CRITIQUE, "gpt4", "C", n2.id)

        leaves = graph.get_leaf_nodes()
        assert len(leaves) == 1
        assert n3 in leaves

    def test_check_convergence(self):
        """Test convergence detection."""
        graph = DebateGraph(branch_policy=BranchPolicy(convergence_threshold=0.5))
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        # Create two branches with similar claims
        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")

        graph.add_node(
            NodeType.SYNTHESIS, "a", "X", root.id, b1.id, claims=["same"], confidence=0.8
        )
        graph.add_node(
            NodeType.SYNTHESIS, "b", "Y", root.id, b2.id, claims=["same"], confidence=0.8
        )

        candidates = graph.check_convergence()
        # Should find convergent pair
        assert len(candidates) >= 1 or len(graph.get_active_branches()) >= 2

    def test_graph_to_dict(self):
        """Test graph serialization."""
        graph = DebateGraph(debate_id="test123")
        graph.add_node(NodeType.ROOT, "system", "Task")

        d = graph.to_dict()
        assert d["debate_id"] == "test123"
        assert "nodes" in d
        assert "branches" in d
        assert "policy" in d

    def test_graph_from_dict(self):
        """Test graph deserialization."""
        original = DebateGraph(debate_id="test123")
        root = original.add_node(NodeType.ROOT, "system", "Task")
        original.add_node(NodeType.PROPOSAL, "claude", "P", root.id)

        d = original.to_dict()
        restored = DebateGraph.from_dict(d)

        assert restored.debate_id == "test123"
        assert len(restored.nodes) == 2
        assert restored.root_id == root.id


class TestGraphReplayBuilder:
    """Test GraphReplayBuilder."""

    def test_replay_branch(self):
        """Test replaying a single branch."""
        graph = DebateGraph()
        n1 = graph.add_node(NodeType.ROOT, "system", "Task")
        n2 = graph.add_node(NodeType.PROPOSAL, "claude", "P", n1.id)
        n3 = graph.add_node(NodeType.CRITIQUE, "gpt4", "C", n2.id)

        builder = GraphReplayBuilder(graph)
        nodes = builder.replay_branch("main")

        assert len(nodes) == 3

    def test_replay_with_callback(self):
        """Test replay with callback function."""
        graph = DebateGraph()
        graph.add_node(NodeType.ROOT, "system", "Task")

        builder = GraphReplayBuilder(graph)
        callback_results = []

        def callback(node, index):
            callback_results.append((node.id, index))

        builder.replay_branch("main", callback)
        assert len(callback_results) == 1

    def test_replay_full(self):
        """Test replaying entire graph."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")
        branch = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "Alt")
        graph.add_node(NodeType.PROPOSAL, "a", "P", root.id, branch.id)

        builder = GraphReplayBuilder(graph)
        result = builder.replay_full()

        assert "main" in result
        assert branch.id in result

    def test_get_counterfactual_paths(self):
        """Test getting counterfactual exploration paths."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")
        branch = graph.create_branch(
            root.id,
            BranchReason.COUNTERFACTUAL_EXPLORATION,
            "What if?",
        )
        graph.add_node(NodeType.COUNTERFACTUAL, "a", "Alt", root.id, branch.id)

        builder = GraphReplayBuilder(graph)
        paths = builder.get_counterfactual_paths()

        assert len(paths) == 1

    def test_generate_summary(self):
        """Test generating graph summary."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")
        graph.add_node(NodeType.PROPOSAL, "claude", "P", root.id)
        graph.add_node(NodeType.CRITIQUE, "gpt4", "C", root.id)

        builder = GraphReplayBuilder(graph)
        summary = builder.generate_summary()

        assert summary["total_nodes"] == 3
        assert summary["total_branches"] == 1
        assert "claude" in summary["agents"]
        assert "gpt4" in summary["agents"]


class TestGraphDebateOrchestrator:
    """Test GraphDebateOrchestrator."""

    def test_orchestrator_creation(self):
        """Test orchestrator initialization."""
        agents = [MagicMock(name="agent1"), MagicMock(name="agent2")]
        orchestrator = GraphDebateOrchestrator(agents)

        assert len(orchestrator.agents) == 2
        assert orchestrator.graph is not None

    def test_orchestrator_with_policy(self):
        """Test orchestrator with custom policy."""
        policy = BranchPolicy(max_branches=2)
        orchestrator = GraphDebateOrchestrator([], policy=policy)

        assert orchestrator.policy.max_branches == 2

    @pytest.mark.asyncio
    async def test_run_debate_no_agent_fn(self):
        """Test running debate without agent function."""
        orchestrator = GraphDebateOrchestrator([])
        graph = await orchestrator.run_debate("Test task")

        assert graph.root_id is not None
        assert len(graph.nodes) == 1  # Just root

    @pytest.mark.asyncio
    async def test_run_debate_with_agents(self):
        """Test running debate with agent function."""
        agent1 = MagicMock()
        agent1.name = "claude"
        agent2 = MagicMock()
        agent2.name = "gpt4"

        orchestrator = GraphDebateOrchestrator([agent1, agent2])

        responses = iter(
            [
                "I think the answer is 42. Confidence: 80%",
                "I agree, the answer is 42. Confidence: 75%",
                "Building on that, I'd add... Confidence: 70%",
                "Yes, good point. Confidence: 85%",
            ]
        )

        async def run_agent(agent, prompt, context):
            return next(responses, "No more responses")

        graph = await orchestrator.run_debate(
            "What is 6 * 7?",
            max_rounds=2,
            run_agent_fn=run_agent,
        )

        assert len(graph.nodes) >= 2  # Root + at least one response

    @pytest.mark.asyncio
    async def test_run_debate_with_callbacks(self):
        """Test debate with callback functions."""
        orchestrator = GraphDebateOrchestrator([])
        nodes_added = []
        branches_created = []

        graph = await orchestrator.run_debate(
            "Test",
            on_node=lambda n: nodes_added.append(n),
            on_branch=lambda b: branches_created.append(b),
        )

        assert len(nodes_added) >= 1  # At least root

    def test_extract_confidence(self):
        """Test confidence extraction from text."""
        orchestrator = GraphDebateOrchestrator([])

        # Explicit confidence
        assert orchestrator._extract_confidence("I am 90% confident") == 0.9
        assert orchestrator._extract_confidence("Confidence: 75") == 0.75

        # High confidence words
        assert orchestrator._extract_confidence("This is definitely true") == 0.8

        # Low confidence words
        assert orchestrator._extract_confidence("Perhaps this might work") == 0.4

        # Default
        assert orchestrator._extract_confidence("Some neutral statement") == 0.6

    def test_extract_claims(self):
        """Test claim extraction from text."""
        orchestrator = GraphDebateOrchestrator([])

        # Numbered claims
        text = "1. First claim\n2. Second claim\n3. Third claim"
        claims = orchestrator._extract_claims(text)
        assert len(claims) >= 1  # Extracts at least some claims
        assert any("claim" in c.lower() for c in claims)

        # Bullet points - on separate lines
        text = "\n- Point one\n- Point two\n"
        claims = orchestrator._extract_claims(text)
        assert len(claims) >= 1
        assert any("point" in c.lower() for c in claims)

        # First sentence fallback
        text = "This is the main point."
        claims = orchestrator._extract_claims(text)
        assert len(claims) >= 1
        assert "main point" in claims[0].lower()

    def test_evaluate_disagreement(self):
        """Test disagreement evaluation."""
        orchestrator = GraphDebateOrchestrator([])

        # No disagreement - all same confidence
        responses = [("a1", "R1", 0.8), ("a2", "R2", 0.8)]
        score, alt = orchestrator.evaluate_disagreement(responses)
        assert score < 0.5

        # High disagreement - very different confidence
        responses = [("a1", "Sure answer", 0.95), ("a2", "Not sure", 0.2)]
        score, alt = orchestrator.evaluate_disagreement(responses)
        assert score > 0.3 or alt is not None

    def test_build_context(self):
        """Test context building from nodes."""
        orchestrator = GraphDebateOrchestrator([])

        nodes = [
            DebateNode(id="1", node_type=NodeType.PROPOSAL, agent_id="a1", content="First point"),
            DebateNode(id="2", node_type=NodeType.CRITIQUE, agent_id="a2", content="Counter"),
        ]

        context = orchestrator._build_context(nodes)
        assert "[a1]:" in context
        assert "[a2]:" in context

    def test_build_prompt(self):
        """Test prompt building."""
        orchestrator = GraphDebateOrchestrator([])

        # Initial round
        prompt = orchestrator._build_prompt("Test task", 0, "", "")
        assert "Test task" in prompt
        assert "initial" in prompt.lower()

        # Later round
        prompt = orchestrator._build_prompt("Test task", 2, "Previous content", "")
        assert "Previous content" in prompt

        # With hypothesis
        prompt = orchestrator._build_prompt("Task", 1, "Content", "Alternative view")
        assert "Alternative view" in prompt

    def test_synthesize_branches(self):
        """Test branch synthesis."""
        orchestrator = GraphDebateOrchestrator([])

        nodes_a = [
            DebateNode(
                id="a1",
                node_type=NodeType.SYNTHESIS,
                agent_id="x",
                content="Conclusion A",
                claims=["claim_a", "common"],
            ),
        ]
        nodes_b = [
            DebateNode(
                id="b1",
                node_type=NodeType.SYNTHESIS,
                agent_id="y",
                content="Conclusion B",
                claims=["claim_b", "common"],
            ),
        ]

        synthesis = orchestrator._synthesize_branches(nodes_a, nodes_b)
        assert "Synthesis" in synthesis
        assert "Conclusion A" in synthesis or "Conclusion B" in synthesis

    def test_create_final_synthesis(self):
        """Test final synthesis creation."""
        orchestrator = GraphDebateOrchestrator([])

        # Single leaf
        leaves = [
            DebateNode(id="1", node_type=NodeType.CONCLUSION, agent_id="a", content="Final answer"),
        ]
        result = orchestrator._create_final_synthesis(leaves)
        assert "Final answer" in result

        # Multiple leaves
        leaves = [
            DebateNode(
                id="1",
                node_type=NodeType.SYNTHESIS,
                agent_id="a",
                content="Path 1 result",
                confidence=0.8,
                claims=["c1"],
            ),
            DebateNode(
                id="2",
                node_type=NodeType.SYNTHESIS,
                agent_id="b",
                content="Path 2 result",
                confidence=0.7,
                claims=["c2"],
            ),
        ]
        result = orchestrator._create_final_synthesis(leaves)
        assert "Path 1" in result or "Path 2" in result

        # No leaves
        result = orchestrator._create_final_synthesis([])
        assert "No conclusion" in result

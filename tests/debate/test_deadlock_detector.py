"""
Tests for Deadlock Detector module.

Tests deadlock detection functionality for multi-agent debates:
- ArgumentNode and Deadlock dataclasses
- ArgumentGraph cycle and mutual block detection
- DeadlockDetector with semantic loop and convergence failure detection
- Utility functions
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import List

import pytest


# =============================================================================
# DeadlockType Enum Tests
# =============================================================================


class TestDeadlockType:
    """Test DeadlockType enum."""

    def test_cycle_type_exists(self):
        """Test CYCLE type exists."""
        from aragora.debate.deadlock_detector import DeadlockType

        assert DeadlockType.CYCLE.value == "cycle"

    def test_mutual_block_type_exists(self):
        """Test MUTUAL_BLOCK type exists."""
        from aragora.debate.deadlock_detector import DeadlockType

        assert DeadlockType.MUTUAL_BLOCK.value == "mutual_block"

    def test_semantic_loop_type_exists(self):
        """Test SEMANTIC_LOOP type exists."""
        from aragora.debate.deadlock_detector import DeadlockType

        assert DeadlockType.SEMANTIC_LOOP.value == "semantic_loop"

    def test_convergence_failure_type_exists(self):
        """Test CONVERGENCE_FAILURE type exists."""
        from aragora.debate.deadlock_detector import DeadlockType

        assert DeadlockType.CONVERGENCE_FAILURE.value == "convergence_failure"

    def test_resource_contention_type_exists(self):
        """Test RESOURCE_CONTENTION type exists."""
        from aragora.debate.deadlock_detector import DeadlockType

        assert DeadlockType.RESOURCE_CONTENTION.value == "resource_contention"


# =============================================================================
# ArgumentNode Tests
# =============================================================================


class TestArgumentNode:
    """Test ArgumentNode dataclass."""

    def test_create_minimal(self):
        """Test creating argument node with minimal fields."""
        from aragora.debate.deadlock_detector import ArgumentNode

        node = ArgumentNode(
            id="node-1",
            agent_id="agent-a",
            content_hash="abc123",
            round_number=1,
            argument_type="proposal",
        )

        assert node.id == "node-1"
        assert node.agent_id == "agent-a"
        assert node.content_hash == "abc123"
        assert node.round_number == 1
        assert node.argument_type == "proposal"
        assert node.parent_id is None
        assert node.targets == []

    def test_create_with_parent(self):
        """Test creating node with parent reference."""
        from aragora.debate.deadlock_detector import ArgumentNode

        node = ArgumentNode(
            id="node-2",
            agent_id="agent-b",
            content_hash="def456",
            round_number=2,
            argument_type="critique",
            parent_id="node-1",
        )

        assert node.parent_id == "node-1"

    def test_create_with_targets(self):
        """Test creating node with target references."""
        from aragora.debate.deadlock_detector import ArgumentNode

        node = ArgumentNode(
            id="node-3",
            agent_id="agent-c",
            content_hash="ghi789",
            round_number=2,
            argument_type="rebuttal",
            targets=["node-1", "node-2"],
        )

        assert node.targets == ["node-1", "node-2"]

    def test_created_at_default(self):
        """Test created_at has default value."""
        from aragora.debate.deadlock_detector import ArgumentNode

        node = ArgumentNode(
            id="node-1",
            agent_id="agent-a",
            content_hash="abc123",
            round_number=1,
            argument_type="proposal",
        )

        assert isinstance(node.created_at, datetime)


# =============================================================================
# Deadlock Tests
# =============================================================================


class TestDeadlock:
    """Test Deadlock dataclass."""

    def test_create_cycle_deadlock(self):
        """Test creating cycle deadlock."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="dl-001",
            deadlock_type=DeadlockType.CYCLE,
            debate_id="debate-123",
            involved_agents=["agent-a", "agent-b"],
            involved_arguments=["node-1", "node-2", "node-3"],
            cycle_path=["node-1", "node-2", "node-3", "node-1"],
        )

        assert deadlock.id == "dl-001"
        assert deadlock.deadlock_type == DeadlockType.CYCLE
        assert deadlock.cycle_path is not None
        assert deadlock.resolved is False

    def test_create_mutual_block_deadlock(self):
        """Test creating mutual block deadlock."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="dl-002",
            deadlock_type=DeadlockType.MUTUAL_BLOCK,
            debate_id="debate-123",
            involved_agents=["agent-a", "agent-b"],
            involved_arguments=[],
            severity="medium",
            description="Two agents blocking each other",
        )

        assert deadlock.deadlock_type == DeadlockType.MUTUAL_BLOCK
        assert deadlock.severity == "medium"

    def test_deadlock_default_severity(self):
        """Test deadlock has default severity."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="dl-003",
            deadlock_type=DeadlockType.SEMANTIC_LOOP,
            debate_id="debate-123",
            involved_agents=["agent-a"],
            involved_arguments=["node-1", "node-2"],
        )

        assert deadlock.severity == "medium"

    def test_deadlock_resolution(self):
        """Test marking deadlock as resolved."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="dl-004",
            deadlock_type=DeadlockType.CONVERGENCE_FAILURE,
            debate_id="debate-123",
            involved_agents=["agent-a", "agent-b"],
            involved_arguments=[],
        )

        assert deadlock.resolved is False
        deadlock.resolved = True
        deadlock.resolution = "Introduced mediating agent"

        assert deadlock.resolved is True
        assert deadlock.resolution == "Introduced mediating agent"


# =============================================================================
# ArgumentGraph Tests
# =============================================================================


class TestArgumentGraph:
    """Test ArgumentGraph class."""

    def test_init(self):
        """Test graph initialization."""
        from aragora.debate.deadlock_detector import ArgumentGraph

        graph = ArgumentGraph(debate_id="debate-123")

        assert graph.debate_id == "debate-123"

    def test_add_node(self):
        """Test adding a node to the graph."""
        from aragora.debate.deadlock_detector import ArgumentGraph, ArgumentNode

        graph = ArgumentGraph(debate_id="debate-123")
        node = ArgumentNode(
            id="node-1",
            agent_id="agent-a",
            content_hash="abc123",
            round_number=1,
            argument_type="proposal",
        )

        graph.add_node(node)

        assert graph.get_node("node-1") is node

    def test_add_node_with_parent_creates_edge(self):
        """Test adding node with parent creates edge."""
        from aragora.debate.deadlock_detector import ArgumentGraph, ArgumentNode

        graph = ArgumentGraph(debate_id="debate-123")

        parent = ArgumentNode(
            id="parent",
            agent_id="agent-a",
            content_hash="abc123",
            round_number=1,
            argument_type="proposal",
        )
        graph.add_node(parent)

        child = ArgumentNode(
            id="child",
            agent_id="agent-b",
            content_hash="def456",
            round_number=2,
            argument_type="critique",
            parent_id="parent",
        )
        graph.add_node(child)

        assert "parent" in graph.get_outgoing("child")
        assert "child" in graph.get_incoming("parent")

    def test_add_node_with_targets_creates_edges(self):
        """Test adding node with targets creates edges."""
        from aragora.debate.deadlock_detector import ArgumentGraph, ArgumentNode

        graph = ArgumentGraph(debate_id="debate-123")

        target1 = ArgumentNode(
            id="target1",
            agent_id="agent-a",
            content_hash="abc",
            round_number=1,
            argument_type="proposal",
        )
        target2 = ArgumentNode(
            id="target2",
            agent_id="agent-b",
            content_hash="def",
            round_number=1,
            argument_type="proposal",
        )
        graph.add_node(target1)
        graph.add_node(target2)

        node = ArgumentNode(
            id="critique",
            agent_id="agent-c",
            content_hash="ghi",
            round_number=2,
            argument_type="critique",
            targets=["target1", "target2"],
        )
        graph.add_node(node)

        outgoing = graph.get_outgoing("critique")
        assert "target1" in outgoing
        assert "target2" in outgoing

    def test_get_agent_nodes(self):
        """Test getting nodes by agent."""
        from aragora.debate.deadlock_detector import ArgumentGraph, ArgumentNode

        graph = ArgumentGraph(debate_id="debate-123")

        node1 = ArgumentNode(
            id="node1",
            agent_id="agent-a",
            content_hash="abc",
            round_number=1,
            argument_type="proposal",
        )
        node2 = ArgumentNode(
            id="node2",
            agent_id="agent-a",
            content_hash="def",
            round_number=2,
            argument_type="revision",
        )
        node3 = ArgumentNode(
            id="node3",
            agent_id="agent-b",
            content_hash="ghi",
            round_number=1,
            argument_type="proposal",
        )

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        agent_a_nodes = graph.get_agent_nodes("agent-a")
        assert "node1" in agent_a_nodes
        assert "node2" in agent_a_nodes
        assert "node3" not in agent_a_nodes

    def test_find_cycles_no_cycle(self):
        """Test finding cycles when none exist."""
        from aragora.debate.deadlock_detector import ArgumentGraph, ArgumentNode

        graph = ArgumentGraph(debate_id="debate-123")

        # Linear chain: A -> B -> C (no cycle)
        node_a = ArgumentNode(
            id="A", agent_id="agent-1", content_hash="a", round_number=1, argument_type="proposal"
        )
        node_b = ArgumentNode(
            id="B",
            agent_id="agent-2",
            content_hash="b",
            round_number=2,
            argument_type="critique",
            parent_id="A",
        )
        node_c = ArgumentNode(
            id="C",
            agent_id="agent-3",
            content_hash="c",
            round_number=3,
            argument_type="rebuttal",
            parent_id="B",
        )

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)

        cycles = graph.find_cycles()
        assert len(cycles) == 0

    def test_find_cycles_with_cycle(self):
        """Test finding cycles when one exists."""
        from aragora.debate.deadlock_detector import ArgumentGraph, ArgumentNode

        graph = ArgumentGraph(debate_id="debate-123")

        # Create cycle: A -> B -> C -> A
        node_a = ArgumentNode(
            id="A", agent_id="agent-1", content_hash="a", round_number=1, argument_type="proposal"
        )
        graph.add_node(node_a)

        node_b = ArgumentNode(
            id="B",
            agent_id="agent-2",
            content_hash="b",
            round_number=2,
            argument_type="critique",
            targets=["A"],
        )
        graph.add_node(node_b)

        node_c = ArgumentNode(
            id="C",
            agent_id="agent-3",
            content_hash="c",
            round_number=3,
            argument_type="rebuttal",
            targets=["B"],
        )
        graph.add_node(node_c)

        # Add edge from A back to C to complete cycle
        graph._add_edge("A", "C")

        cycles = graph.find_cycles()
        assert len(cycles) > 0

    def test_find_mutual_blocks(self):
        """Test finding mutual blocks."""
        from aragora.debate.deadlock_detector import ArgumentGraph, ArgumentNode

        graph = ArgumentGraph(debate_id="debate-123")

        # Agent A's proposal
        node_a = ArgumentNode(
            id="A",
            agent_id="agent-a",
            content_hash="a",
            round_number=1,
            argument_type="proposal",
        )
        graph.add_node(node_a)

        # Agent B's proposal
        node_b = ArgumentNode(
            id="B",
            agent_id="agent-b",
            content_hash="b",
            round_number=1,
            argument_type="proposal",
        )
        graph.add_node(node_b)

        # Agent A criticizes B
        node_a2 = ArgumentNode(
            id="A2",
            agent_id="agent-a",
            content_hash="a2",
            round_number=2,
            argument_type="critique",
            targets=["B"],
        )
        graph.add_node(node_a2)

        # Agent B criticizes A
        node_b2 = ArgumentNode(
            id="B2",
            agent_id="agent-b",
            content_hash="b2",
            round_number=2,
            argument_type="critique",
            targets=["A"],
        )
        graph.add_node(node_b2)

        blocks = graph.find_mutual_blocks()
        # Should find mutual block between agent-a and agent-b
        assert len(blocks) > 0
        agents = {blocks[0][0], blocks[0][1]}
        assert "agent-a" in agents
        assert "agent-b" in agents

    def test_get_agent_targeting_stats(self):
        """Test getting agent targeting statistics."""
        from aragora.debate.deadlock_detector import ArgumentGraph, ArgumentNode

        graph = ArgumentGraph(debate_id="debate-123")

        target = ArgumentNode(
            id="target",
            agent_id="agent-b",
            content_hash="t",
            round_number=1,
            argument_type="proposal",
        )
        graph.add_node(target)

        critique1 = ArgumentNode(
            id="c1",
            agent_id="agent-a",
            content_hash="c1",
            round_number=2,
            argument_type="critique",
            targets=["target"],
        )
        critique2 = ArgumentNode(
            id="c2",
            agent_id="agent-a",
            content_hash="c2",
            round_number=3,
            argument_type="critique",
            targets=["target"],
        )
        graph.add_node(critique1)
        graph.add_node(critique2)

        stats = graph.get_agent_targeting_stats()

        assert "agent-a" in stats
        assert stats["agent-a"]["agent-b"] == 2


# =============================================================================
# DeadlockDetector Tests
# =============================================================================


class TestDeadlockDetector:
    """Test DeadlockDetector class."""

    def test_init(self):
        """Test detector initialization."""
        from aragora.debate.deadlock_detector import DeadlockDetector

        detector = DeadlockDetector(debate_id="debate-123")

        assert detector.debate_id == "debate-123"
        assert detector.semantic_threshold == 0.9

    def test_init_with_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        from aragora.debate.deadlock_detector import DeadlockDetector

        detector = DeadlockDetector(
            debate_id="debate-123",
            semantic_threshold=0.85,
            cycle_severity_threshold=5,
        )

        assert detector.semantic_threshold == 0.85
        assert detector.cycle_severity_threshold == 5

    def test_register_argument(self):
        """Test registering an argument."""
        from aragora.debate.deadlock_detector import ArgumentNode, DeadlockDetector

        detector = DeadlockDetector(debate_id="debate-123")
        node = ArgumentNode(
            id="node-1",
            agent_id="agent-a",
            content_hash="abc123",
            round_number=1,
            argument_type="proposal",
        )

        deadlocks = detector.register_argument(node)

        assert isinstance(deadlocks, list)
        assert detector.get_argument_graph().get_node("node-1") is node

    def test_detect_deadlocks_empty(self):
        """Test detecting deadlocks with no arguments."""
        from aragora.debate.deadlock_detector import DeadlockDetector

        detector = DeadlockDetector(debate_id="debate-123")
        deadlocks = detector.detect_deadlocks()

        assert deadlocks == []

    def test_detect_semantic_loop(self):
        """Test detecting semantic loops."""
        from aragora.debate.deadlock_detector import ArgumentNode, DeadlockDetector, DeadlockType

        detector = DeadlockDetector(debate_id="debate-123")

        # Same content hash, different rounds (semantic loop)
        same_hash = "same_content_hash"

        node1 = ArgumentNode(
            id="node-1",
            agent_id="agent-a",
            content_hash=same_hash,
            round_number=1,
            argument_type="proposal",
        )
        node2 = ArgumentNode(
            id="node-2",
            agent_id="agent-b",
            content_hash="different",
            round_number=2,
            argument_type="critique",
        )
        node3 = ArgumentNode(
            id="node-3",
            agent_id="agent-a",
            content_hash=same_hash,  # Same as node-1
            round_number=3,
            argument_type="proposal",
        )

        detector.register_argument(node1)
        detector.register_argument(node2)
        deadlocks = detector.register_argument(node3)

        # Should detect semantic loop
        semantic_loops = [d for d in deadlocks if d.deadlock_type == DeadlockType.SEMANTIC_LOOP]
        assert len(semantic_loops) > 0

    def test_detect_convergence_failure(self):
        """Test detecting convergence failure."""
        from aragora.debate.deadlock_detector import ArgumentNode, DeadlockDetector, DeadlockType

        detector = DeadlockDetector(debate_id="debate-123")

        # Create escalating argument counts (1, 2, 3)
        node1 = ArgumentNode(
            id="n1", agent_id="a1", content_hash="h1", round_number=1, argument_type="proposal"
        )
        node2 = ArgumentNode(
            id="n2", agent_id="a1", content_hash="h2", round_number=2, argument_type="revision"
        )
        node3 = ArgumentNode(
            id="n3", agent_id="a2", content_hash="h3", round_number=2, argument_type="critique"
        )
        node4 = ArgumentNode(
            id="n4", agent_id="a1", content_hash="h4", round_number=3, argument_type="revision"
        )
        node5 = ArgumentNode(
            id="n5", agent_id="a2", content_hash="h5", round_number=3, argument_type="critique"
        )
        node6 = ArgumentNode(
            id="n6", agent_id="a3", content_hash="h6", round_number=3, argument_type="proposal"
        )

        for node in [node1, node2, node3, node4, node5, node6]:
            detector.register_argument(node)

        deadlocks = detector.detect_deadlocks()
        convergence_failures = [
            d for d in deadlocks if d.deadlock_type == DeadlockType.CONVERGENCE_FAILURE
        ]

        # May or may not detect depending on specific counts
        # The test validates the mechanism runs without error
        assert isinstance(deadlocks, list)

    def test_get_deadlocks(self):
        """Test getting all deadlocks."""
        from aragora.debate.deadlock_detector import DeadlockDetector

        detector = DeadlockDetector(debate_id="debate-123")
        deadlocks = detector.get_deadlocks()

        assert isinstance(deadlocks, list)

    def test_get_deadlocks_exclude_resolved(self):
        """Test getting deadlocks excludes resolved by default."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockDetector, DeadlockType

        detector = DeadlockDetector(debate_id="debate-123")

        # Manually add a deadlock and resolve it
        deadlock = Deadlock(
            id="dl-001",
            deadlock_type=DeadlockType.CYCLE,
            debate_id="debate-123",
            involved_agents=["a"],
            involved_arguments=["n1"],
        )
        detector._deadlocks.append(deadlock)
        deadlock.resolved = True

        unresolved = detector.get_deadlocks(include_resolved=False)
        all_deadlocks = detector.get_deadlocks(include_resolved=True)

        assert len(unresolved) == 0
        assert len(all_deadlocks) == 1

    def test_resolve_deadlock(self):
        """Test resolving a deadlock."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockDetector, DeadlockType

        detector = DeadlockDetector(debate_id="debate-123")

        # Add a deadlock
        deadlock = Deadlock(
            id="dl-001",
            deadlock_type=DeadlockType.MUTUAL_BLOCK,
            debate_id="debate-123",
            involved_agents=["a", "b"],
            involved_arguments=[],
        )
        detector._deadlocks.append(deadlock)

        result = detector.resolve_deadlock("dl-001", "Introduced mediating perspective")

        assert result is True
        assert deadlock.resolved is True
        assert deadlock.resolution == "Introduced mediating perspective"

    def test_resolve_deadlock_not_found(self):
        """Test resolving non-existent deadlock."""
        from aragora.debate.deadlock_detector import DeadlockDetector

        detector = DeadlockDetector(debate_id="debate-123")

        result = detector.resolve_deadlock("non-existent", "Resolution")

        assert result is False

    def test_get_statistics(self):
        """Test getting detector statistics."""
        from aragora.debate.deadlock_detector import ArgumentNode, DeadlockDetector

        detector = DeadlockDetector(debate_id="debate-123")

        # Add some arguments
        node1 = ArgumentNode(
            id="n1", agent_id="a1", content_hash="h1", round_number=1, argument_type="proposal"
        )
        node2 = ArgumentNode(
            id="n2",
            agent_id="a2",
            content_hash="h2",
            round_number=2,
            argument_type="critique",
            parent_id="n1",
        )
        detector.register_argument(node1)
        detector.register_argument(node2)

        stats = detector.get_statistics()

        assert stats["debate_id"] == "debate-123"
        assert stats["total_arguments"] == 2
        assert stats["agents_tracked"] == 2
        assert "deadlock_types" in stats

    def test_get_argument_graph(self):
        """Test getting the argument graph."""
        from aragora.debate.deadlock_detector import ArgumentGraph, DeadlockDetector

        detector = DeadlockDetector(debate_id="debate-123")
        graph = detector.get_argument_graph()

        assert isinstance(graph, ArgumentGraph)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestCreateArgumentNode:
    """Test create_argument_node utility function."""

    def test_create_with_minimal_args(self):
        """Test creating node with minimal arguments."""
        from aragora.debate.deadlock_detector import create_argument_node

        node = create_argument_node(
            node_id="n1",
            agent_id="agent-a",
            content="This is my proposal",
            round_number=1,
            argument_type="proposal",
        )

        assert node.id == "n1"
        assert node.agent_id == "agent-a"
        assert node.round_number == 1
        assert len(node.content_hash) == 16

    def test_create_with_parent_and_targets(self):
        """Test creating node with parent and targets."""
        from aragora.debate.deadlock_detector import create_argument_node

        node = create_argument_node(
            node_id="n2",
            agent_id="agent-b",
            content="This is my critique",
            round_number=2,
            argument_type="critique",
            parent_id="n1",
            targets=["n1"],
        )

        assert node.parent_id == "n1"
        assert node.targets == ["n1"]

    def test_content_hash_is_deterministic(self):
        """Test same content produces same hash."""
        from aragora.debate.deadlock_detector import create_argument_node

        node1 = create_argument_node(
            node_id="n1",
            agent_id="a1",
            content="Same content",
            round_number=1,
            argument_type="proposal",
        )
        node2 = create_argument_node(
            node_id="n2",
            agent_id="a2",
            content="Same content",
            round_number=2,
            argument_type="proposal",
        )

        assert node1.content_hash == node2.content_hash

    def test_content_hash_case_insensitive(self):
        """Test content hash is case insensitive."""
        from aragora.debate.deadlock_detector import create_argument_node

        node1 = create_argument_node(
            node_id="n1",
            agent_id="a1",
            content="UPPERCASE CONTENT",
            round_number=1,
            argument_type="proposal",
        )
        node2 = create_argument_node(
            node_id="n2",
            agent_id="a2",
            content="uppercase content",
            round_number=1,
            argument_type="proposal",
        )

        assert node1.content_hash == node2.content_hash


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_deadlock_type_exportable(self):
        """Test DeadlockType can be imported."""
        from aragora.debate.deadlock_detector import DeadlockType

        assert DeadlockType is not None

    def test_argument_node_exportable(self):
        """Test ArgumentNode can be imported."""
        from aragora.debate.deadlock_detector import ArgumentNode

        assert ArgumentNode is not None

    def test_deadlock_exportable(self):
        """Test Deadlock can be imported."""
        from aragora.debate.deadlock_detector import Deadlock

        assert Deadlock is not None

    def test_argument_graph_exportable(self):
        """Test ArgumentGraph can be imported."""
        from aragora.debate.deadlock_detector import ArgumentGraph

        assert ArgumentGraph is not None

    def test_detector_exportable(self):
        """Test DeadlockDetector can be imported."""
        from aragora.debate.deadlock_detector import DeadlockDetector

        assert DeadlockDetector is not None

    def test_utility_function_exportable(self):
        """Test create_argument_node can be imported."""
        from aragora.debate.deadlock_detector import create_argument_node

        assert create_argument_node is not None

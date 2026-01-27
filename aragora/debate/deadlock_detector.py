"""
Deadlock Detector for Multi-Agent Debates.

Detects circular dependencies and deadlock patterns in debate arguments:
- Argument graph cycle detection
- Mutual blocking detection (A waits for B, B waits for A)
- Converging disagreement loops
- Semantic loop detection (repeated argument patterns)

Inspired by gastown's deadlock detection in distributed work coordination.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DeadlockType(str, Enum):
    """Types of deadlock patterns."""

    CYCLE = "cycle"  # Circular reference in arguments
    MUTUAL_BLOCK = "mutual_block"  # Two agents blocking each other
    SEMANTIC_LOOP = "semantic_loop"  # Repeated argument patterns
    CONVERGENCE_FAILURE = "convergence_failure"  # No progress toward consensus
    RESOURCE_CONTENTION = "resource_contention"  # Competing for same resource


@dataclass
class ArgumentNode:
    """A node in the argument graph representing a proposal, critique, or rebuttal."""

    id: str
    agent_id: str
    content_hash: str  # For semantic comparison
    round_number: int
    argument_type: str  # "proposal", "critique", "rebuttal", "revision"
    parent_id: Optional[str] = None  # What this argument responds to
    targets: List[str] = field(default_factory=list)  # Arguments this criticizes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Deadlock:
    """A detected deadlock situation."""

    id: str
    deadlock_type: DeadlockType
    debate_id: str
    involved_agents: List[str]
    involved_arguments: List[str]
    cycle_path: Optional[List[str]] = None  # For cycle-type deadlocks
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    resolution_hint: Optional[str] = None
    resolved: bool = False
    resolution: Optional[str] = None


class ArgumentGraph:
    """
    Graph structure for tracking argument relationships.

    Maintains directed edges representing argument-response relationships:
    - proposal -> critique (critique targets proposal)
    - critique -> rebuttal (rebuttal responds to critique)
    - revision -> previous_proposal (revision supersedes)
    """

    def __init__(self, debate_id: str):
        """Initialize the argument graph."""
        self.debate_id = debate_id
        self._nodes: Dict[str, ArgumentNode] = {}
        self._edges: Dict[str, Set[str]] = defaultdict(set)  # from -> [to]
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # to -> [from]
        self._agent_nodes: Dict[str, Set[str]] = defaultdict(set)  # agent -> nodes

    def add_node(self, node: ArgumentNode) -> None:
        """Add an argument node to the graph."""
        self._nodes[node.id] = node
        self._agent_nodes[node.agent_id].add(node.id)

        # Add edges based on relationships
        if node.parent_id and node.parent_id in self._nodes:
            self._add_edge(node.id, node.parent_id)

        for target_id in node.targets:
            if target_id in self._nodes:
                self._add_edge(node.id, target_id)

    def _add_edge(self, from_id: str, to_id: str) -> None:
        """Add a directed edge."""
        self._edges[from_id].add(to_id)
        self._reverse_edges[to_id].add(from_id)

    def get_node(self, node_id: str) -> Optional[ArgumentNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> Set[str]:
        """Get nodes that this node points to."""
        return self._edges.get(node_id, set())

    def get_incoming(self, node_id: str) -> Set[str]:
        """Get nodes that point to this node."""
        return self._reverse_edges.get(node_id, set())

    def get_agent_nodes(self, agent_id: str) -> Set[str]:
        """Get all nodes belonging to an agent."""
        return self._agent_nodes.get(agent_id, set())

    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in the argument graph.

        Uses DFS-based cycle detection.

        Returns:
            List of cycles, each cycle is a list of node IDs
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in self._edges.get(node_id, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle - extract it
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node_id)
            return False

        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)

        return cycles

    def find_mutual_blocks(self) -> List[Tuple[str, str]]:
        """
        Find pairs of agents that are mutually blocking each other.

        A mutual block occurs when:
        - Agent A's latest argument targets Agent B
        - Agent B's latest argument targets Agent A
        - Neither has made progress since

        Returns:
            List of (agent_a, agent_b) pairs
        """
        mutual_blocks = []

        # Get latest argument per agent
        latest_per_agent: Dict[str, ArgumentNode] = {}
        for node in self._nodes.values():
            if (
                node.agent_id not in latest_per_agent
                or node.created_at > latest_per_agent[node.agent_id].created_at
            ):
                latest_per_agent[node.agent_id] = node

        # Check for mutual targeting
        agent_ids = list(latest_per_agent.keys())
        for i, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[i + 1 :]:
                node_a = latest_per_agent[agent_a]
                node_b = latest_per_agent[agent_b]

                # Check if A targets B's nodes
                a_targets_b = any(
                    target in self._agent_nodes.get(agent_b, set()) for target in node_a.targets
                )
                # Check if B targets A's nodes
                b_targets_a = any(
                    target in self._agent_nodes.get(agent_a, set()) for target in node_b.targets
                )

                if a_targets_b and b_targets_a:
                    # Check if these are the most recent arguments (potential block)
                    mutual_blocks.append((agent_a, agent_b))

        return mutual_blocks

    def get_agent_targeting_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics on which agents target which other agents.

        Returns:
            Dict of {agent_id: {target_agent_id: count}}
        """
        stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for node in self._nodes.values():
            for target_id in node.targets:
                target_node = self._nodes.get(target_id)
                if target_node:
                    stats[node.agent_id][target_node.agent_id] += 1

        return {k: dict(v) for k, v in stats.items()}


class DeadlockDetector:
    """
    Detects deadlock patterns in multi-agent debates.

    Monitors the argument graph and debate state for patterns that
    indicate the debate is stuck in an unproductive loop.

    Usage:
        detector = DeadlockDetector(debate_id="debate-123")

        # Register arguments as they occur
        detector.register_argument(ArgumentNode(...))

        # Check for deadlocks
        deadlocks = detector.detect_deadlocks()

        # Get specific deadlock types
        cycles = detector.find_argument_cycles()
        blocks = detector.find_mutual_blocks()
    """

    def __init__(
        self,
        debate_id: str,
        semantic_threshold: float = 0.9,
        cycle_severity_threshold: int = 3,
    ):
        """
        Initialize the deadlock detector.

        Args:
            debate_id: ID of the debate
            semantic_threshold: Similarity threshold for semantic loop detection
            cycle_severity_threshold: Number of nodes in cycle for high severity
        """
        self.debate_id = debate_id
        self.semantic_threshold = semantic_threshold
        self.cycle_severity_threshold = cycle_severity_threshold

        self._graph = ArgumentGraph(debate_id)
        self._deadlocks: List[Deadlock] = []
        self._content_history: Dict[str, List[str]] = defaultdict(list)  # hash -> [node_ids]
        self._round_argument_counts: Dict[int, int] = defaultdict(int)
        self._deadlock_counter = 0

    def register_argument(self, node: ArgumentNode) -> List[Deadlock]:
        """
        Register a new argument and check for new deadlocks.

        Args:
            node: The argument node to register

        Returns:
            Any new deadlocks detected
        """
        self._graph.add_node(node)
        self._content_history[node.content_hash].append(node.id)
        self._round_argument_counts[node.round_number] += 1

        # Check for deadlocks after registration
        return self.detect_deadlocks()

    def detect_deadlocks(self) -> List[Deadlock]:
        """
        Run all deadlock detection checks.

        Returns:
            List of newly detected deadlocks
        """
        new_deadlocks = []

        # Check for cycles
        new_deadlocks.extend(self._detect_cycles())

        # Check for mutual blocks
        new_deadlocks.extend(self._detect_mutual_blocks())

        # Check for semantic loops
        new_deadlocks.extend(self._detect_semantic_loops())

        # Check for convergence failures
        new_deadlocks.extend(self._detect_convergence_failures())

        return new_deadlocks

    def _detect_cycles(self) -> List[Deadlock]:
        """Detect cycles in the argument graph."""
        deadlocks = []
        cycles = self._graph.find_cycles()

        for cycle in cycles:
            # Check if this cycle is already tracked
            cycle_key = tuple(sorted(cycle))
            if any(
                d.cycle_path and tuple(sorted(d.cycle_path)) == cycle_key
                for d in self._deadlocks
                if not d.resolved
            ):
                continue

            # Determine involved agents
            involved_agents = set()
            for node_id in cycle:
                node = self._graph.get_node(node_id)
                if node:
                    involved_agents.add(node.agent_id)

            severity = "high" if len(cycle) >= self.cycle_severity_threshold else "medium"

            deadlock = Deadlock(
                id=self._generate_id(),
                deadlock_type=DeadlockType.CYCLE,
                debate_id=self.debate_id,
                involved_agents=list(involved_agents),
                involved_arguments=cycle,
                cycle_path=cycle,
                severity=severity,
                description=f"Circular argument chain detected: {' -> '.join(cycle[:5])}{'...' if len(cycle) > 5 else ''}",
                resolution_hint="Introduce new perspective or evidence to break the cycle",
            )
            deadlocks.append(deadlock)
            self._deadlocks.append(deadlock)

        return deadlocks

    def _detect_mutual_blocks(self) -> List[Deadlock]:
        """Detect mutual blocking between agents."""
        deadlocks = []
        blocks = self._graph.find_mutual_blocks()

        for agent_a, agent_b in blocks:
            # Check if already tracked
            if any(
                d.deadlock_type == DeadlockType.MUTUAL_BLOCK
                and set(d.involved_agents) == {agent_a, agent_b}
                for d in self._deadlocks
                if not d.resolved
            ):
                continue

            deadlock = Deadlock(
                id=self._generate_id(),
                deadlock_type=DeadlockType.MUTUAL_BLOCK,
                debate_id=self.debate_id,
                involved_agents=[agent_a, agent_b],
                involved_arguments=[],
                severity="medium",
                description=f"Agents {agent_a} and {agent_b} are mutually blocking each other",
                resolution_hint="Introduce a mediating agent or shift focus to common ground",
            )
            deadlocks.append(deadlock)
            self._deadlocks.append(deadlock)

        return deadlocks

    def _detect_semantic_loops(self) -> List[Deadlock]:
        """Detect semantic loops (repeated argument patterns)."""
        deadlocks = []

        # Find content hashes with multiple occurrences
        for content_hash, node_ids in self._content_history.items():
            if len(node_ids) < 2:
                continue

            # Check if across different rounds (indicates repetition)
            nodes = [self._graph.get_node(nid) for nid in node_ids if self._graph.get_node(nid)]
            rounds = {n.round_number for n in nodes if n}

            if len(rounds) >= 2:
                agents = {n.agent_id for n in nodes if n}

                # Check if already tracked
                if any(
                    d.deadlock_type == DeadlockType.SEMANTIC_LOOP
                    and set(d.involved_arguments) == set(node_ids)
                    for d in self._deadlocks
                    if not d.resolved
                ):
                    continue

                deadlock = Deadlock(
                    id=self._generate_id(),
                    deadlock_type=DeadlockType.SEMANTIC_LOOP,
                    debate_id=self.debate_id,
                    involved_agents=list(agents),
                    involved_arguments=node_ids,
                    severity="medium" if len(node_ids) == 2 else "high",
                    description=f"Similar arguments repeated across rounds {sorted(rounds)}",
                    resolution_hint="Request novel perspectives or additional evidence",
                )
                deadlocks.append(deadlock)
                self._deadlocks.append(deadlock)

        return deadlocks

    def _detect_convergence_failures(self) -> List[Deadlock]:
        """Detect when debate is failing to converge."""
        deadlocks = []

        # Need at least 3 rounds to detect convergence failure
        if len(self._round_argument_counts) < 3:
            return deadlocks

        # Check if argument count is increasing (escalating conflict)
        sorted_rounds = sorted(self._round_argument_counts.items())
        recent_counts = [count for _, count in sorted_rounds[-3:]]

        if all(recent_counts[i] <= recent_counts[i + 1] for i in range(len(recent_counts) - 1)):
            # Arguments not decreasing - potential convergence failure
            if any(
                d.deadlock_type == DeadlockType.CONVERGENCE_FAILURE
                for d in self._deadlocks
                if not d.resolved
            ):
                return deadlocks

            # Get all agents involved in recent rounds
            agents = set()
            for node in self._graph._nodes.values():
                if node.round_number >= sorted_rounds[-3][0]:
                    agents.add(node.agent_id)

            deadlock = Deadlock(
                id=self._generate_id(),
                deadlock_type=DeadlockType.CONVERGENCE_FAILURE,
                debate_id=self.debate_id,
                involved_agents=list(agents),
                involved_arguments=[],
                severity="low",
                description="Debate is not converging: argument count increasing across rounds",
                resolution_hint="Consider early voting or introducing convergence pressure",
            )
            deadlocks.append(deadlock)
            self._deadlocks.append(deadlock)

        return deadlocks

    def _generate_id(self) -> str:
        """Generate a unique deadlock ID."""
        self._deadlock_counter += 1
        return f"deadlock-{self.debate_id[:8]}-{self._deadlock_counter:04d}"

    def get_deadlocks(self, include_resolved: bool = False) -> List[Deadlock]:
        """Get all detected deadlocks."""
        if include_resolved:
            return self._deadlocks
        return [d for d in self._deadlocks if not d.resolved]

    def resolve_deadlock(self, deadlock_id: str, resolution: str) -> bool:
        """Mark a deadlock as resolved."""
        for deadlock in self._deadlocks:
            if deadlock.id == deadlock_id:
                deadlock.resolved = True
                deadlock.resolution = resolution
                logger.info(f"Deadlock {deadlock_id} resolved: {resolution}")
                return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "debate_id": self.debate_id,
            "total_arguments": len(self._graph._nodes),
            "total_edges": sum(len(edges) for edges in self._graph._edges.values()),
            "total_deadlocks": len(self._deadlocks),
            "unresolved_deadlocks": len([d for d in self._deadlocks if not d.resolved]),
            "deadlock_types": {
                dt.value: len([d for d in self._deadlocks if d.deadlock_type == dt])
                for dt in DeadlockType
            },
            "rounds_tracked": len(self._round_argument_counts),
            "agents_tracked": len(self._graph._agent_nodes),
        }

    def get_argument_graph(self) -> ArgumentGraph:
        """Get the underlying argument graph."""
        return self._graph


# Convenience function
def create_argument_node(
    node_id: str,
    agent_id: str,
    content: str,
    round_number: int,
    argument_type: str,
    parent_id: Optional[str] = None,
    targets: Optional[List[str]] = None,
) -> ArgumentNode:
    """Create an argument node with content hash."""
    import hashlib

    content_hash = hashlib.sha256(content.lower().encode()).hexdigest()[:16]
    return ArgumentNode(
        id=node_id,
        agent_id=agent_id,
        content_hash=content_hash,
        round_number=round_number,
        argument_type=argument_type,
        parent_id=parent_id,
        targets=targets or [],
    )

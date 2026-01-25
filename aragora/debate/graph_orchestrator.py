"""
Graph Debate Orchestrator - Runs graph-based debates with branching.

Orchestrates non-linear debates where agents can explore alternative
reasoning paths in parallel and merge when consensus is reached.

Usage:
    from aragora.debate.graph_orchestrator import (
        GraphDebateOrchestrator,
        BranchPolicy,
        MergeStrategy,
    )

    policy = BranchPolicy(min_disagreement=0.5, max_branches=3)
    orchestrator = GraphDebateOrchestrator(agents=agents, policy=policy)

    graph = await orchestrator.run_debate(
        task="Debate topic",
        max_rounds=5,
        run_agent_fn=run_agent,
    )
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Protocol

from aragora.debate.graph import (
    BranchReason,
    DebateGraph,
    DebateNode,
    MergeStrategy,
    NodeType,
)

logger = logging.getLogger(__name__)


# Re-export for convenience
__all__ = [
    "GraphDebateOrchestrator",
    "BranchPolicy",
    "MergeStrategy",
    "GraphNode",
    "GraphDebateResult",
    "DebateGraph",
    "GraphBranch",
]


class Agent(Protocol):
    """Protocol for agents that can generate responses."""

    name: str

    async def generate(self, prompt: str, context: Any = None) -> str: ...


@dataclass
class GraphNode:
    """
    A node in the debate graph.

    Simplified interface for graph debate nodes.
    """

    id: str
    content: str
    agent: str
    round_num: int
    branch_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    node_type: str = "proposal"
    confidence: float = 0.0
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_debate_node(cls, node: DebateNode, round_num: int = 0) -> "GraphNode":
        """Create a GraphNode from a DebateNode."""
        return cls(
            id=node.id,
            content=node.content,
            agent=node.agent_id,
            round_num=round_num,
            branch_id=node.branch_id or "main",
            timestamp=node.timestamp,
            node_type=node.node_type.value,
            confidence=node.confidence,
            parent_ids=node.parent_ids,
            child_ids=node.child_ids,
            metadata=node.metadata,
        )

    def to_dict(self) -> dict:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "agent": self.agent,
            "round_num": self.round_num,
            "branch_id": self.branch_id,
            "timestamp": self.timestamp.isoformat(),
            "node_type": self.node_type,
            "confidence": self.confidence,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "metadata": self.metadata,
        }


@dataclass
class BranchPolicy:
    """
    Policy for when to create and merge branches.

    Attributes:
        min_disagreement: Minimum disagreement score to trigger branching (0-1)
        max_branches: Maximum number of concurrent branches allowed
        auto_merge: Whether to automatically merge converging branches
        merge_strategy: Strategy for merging branches
        min_confidence_to_branch: Minimum confidence in alternative for branching
        convergence_threshold: Similarity threshold for auto-merging (0-1)
    """

    min_disagreement: float = 0.6
    max_branches: int = 4
    auto_merge: bool = True
    merge_strategy: MergeStrategy = field(default=MergeStrategy.SYNTHESIS)
    min_confidence_to_branch: float = 0.3
    convergence_threshold: float = 0.8

    def should_branch(
        self,
        disagreement: float,
        current_branches: int,
        alternative_confidence: float = 0.0,
    ) -> tuple[bool, Optional[BranchReason]]:
        """Determine if a branch should be created."""
        if current_branches >= self.max_branches:
            return False, None

        if disagreement >= self.min_disagreement:
            return True, BranchReason.HIGH_DISAGREEMENT

        if alternative_confidence >= self.min_confidence_to_branch:
            return True, BranchReason.ALTERNATIVE_APPROACH

        return False, None


@dataclass
class GraphBranch:
    """A branch in the graph debate."""

    id: str
    name: str
    start_node_id: str
    hypothesis: str = ""
    is_active: bool = True
    is_merged: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_node_id": self.start_node_id,
            "hypothesis": self.hypothesis,
            "is_active": self.is_active,
            "is_merged": self.is_merged,
        }


@dataclass
class GraphDebateResult:
    """Result of a graph debate."""

    id: str
    task: str
    nodes: list[GraphNode]
    branches: list[GraphBranch]
    winner: Optional[str] = None
    synthesis: Optional[str] = None
    total_rounds: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "nodes": [n.to_dict() for n in self.nodes],
            "branches": [b.to_dict() for b in self.branches],
            "winner": self.winner,
            "synthesis": self.synthesis,
            "total_rounds": self.total_rounds,
            "metadata": self.metadata,
        }


class GraphDebateOrchestrator:
    """
    Orchestrator for graph-based debates with branching.

    Runs debates that can branch when agents disagree strongly,
    exploring alternative reasoning paths in parallel and merging
    when consensus is reached.
    """

    def __init__(
        self,
        agents: list[Agent],
        policy: Optional[BranchPolicy] = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ):
        """
        Initialize the graph debate orchestrator.

        Args:
            agents: List of agents participating in the debate
            policy: Branch policy controlling when to branch/merge
            event_callback: Optional callback for debate events
        """
        self.agents = agents
        self.policy = policy or BranchPolicy()
        self.event_callback = event_callback
        self._graph: Optional[DebateGraph] = None

    async def run_debate(
        self,
        task: str,
        max_rounds: int = 5,
        run_agent_fn: Optional[Callable[[Agent, str, Any], Any]] = None,
        context: Optional[dict] = None,
        event_emitter: Optional[Any] = None,
    ) -> GraphDebateResult:
        """
        Run a graph-based debate.

        Args:
            task: The debate topic/task
            max_rounds: Maximum number of debate rounds
            run_agent_fn: Function to run agents (async)
            context: Optional context for the debate
            event_emitter: Optional event emitter for real-time updates

        Returns:
            GraphDebateResult with all nodes and branches
        """
        # Store event_emitter if provided
        if event_emitter is not None:
            self.event_callback = lambda t, d: event_emitter.emit({"type": t, **d})

        debate_id = str(uuid.uuid4())[:8]
        self._graph = DebateGraph(debate_id=debate_id)

        self._emit_event("debate_start", {"debate_id": debate_id, "task": task})

        # Run debate rounds
        current_round = 0
        for round_num in range(max_rounds):
            current_round = round_num + 1
            self._emit_event("round_start", {"round": current_round})

            # Get responses from all agents
            responses = await self._collect_responses(task, round_num, run_agent_fn, context)

            # Add responses as nodes
            for agent, response in responses.items():
                self._add_response_node(agent, response, round_num)

            # Check for branching conditions
            if current_round > 1:
                await self._check_branching(responses, round_num)

            # Check for convergence and auto-merge
            if self.policy.auto_merge:
                await self._check_convergence_and_merge(run_agent_fn, context)

            self._emit_event("round_end", {"round": current_round})

        # Final synthesis
        synthesis = await self._generate_synthesis(task, run_agent_fn, context)

        self._emit_event("debate_end", {"debate_id": debate_id})

        return self._build_result(debate_id, task, synthesis, current_round)

    async def _collect_responses(
        self,
        task: str,
        round_num: int,
        run_agent_fn: Optional[Callable],
        context: Optional[dict],
    ) -> dict[str, str]:
        """Collect responses from all agents."""
        responses: dict[str, str] = {}

        for agent in self.agents:
            prompt = self._build_prompt(task, round_num, context)

            if run_agent_fn:
                response = await run_agent_fn(agent, prompt, context)
            else:
                response = await agent.generate(prompt, context)

            responses[agent.name] = response
            self._emit_event(
                "agent_response",
                {
                    "agent": agent.name,
                    "round": round_num + 1,
                    "response_length": len(response),
                },
            )

        return responses

    def _build_prompt(self, task: str, round_num: int, context: Optional[dict]) -> str:
        """Build prompt for an agent."""
        if round_num == 0:
            return f"Task: {task}\n\nProvide your initial proposal."

        # Include previous responses in context
        history = self._get_debate_history()
        return f"Task: {task}\n\nPrevious discussion:\n{history}\n\nProvide your response."

    def _get_debate_history(self) -> str:
        """Get formatted debate history."""
        if not self._graph:
            return ""

        history_parts = []
        for node in self._graph.nodes.values():
            history_parts.append(f"[{node.agent_id}]: {node.content[:200]}...")

        return "\n".join(history_parts[-10:])  # Last 10 messages

    def _add_response_node(self, agent: str, response: str, round_num: int) -> None:
        """Add a response as a node in the graph."""
        if not self._graph:
            return

        # Get parent node (previous response from same agent or last node)
        parent_id = None
        if self._graph.nodes:
            # Find the latest node
            latest_nodes = sorted(
                self._graph.nodes.values(),
                key=lambda n: n.timestamp,
                reverse=True,
            )
            if latest_nodes:
                parent_id = latest_nodes[0].id

        # Determine node type
        node_type = NodeType.PROPOSAL if round_num == 0 else NodeType.CRITIQUE

        # Add node
        self._graph.add_node(
            node_type=node_type,
            agent_id=agent,
            content=response,
            parent_id=parent_id,
            metadata={"round": round_num + 1},
        )

    async def _check_branching(self, responses: dict[str, str], round_num: int) -> None:
        """Check if branching is needed based on disagreement."""
        if not self._graph:
            return

        # Calculate disagreement between agents
        disagreement = self._calculate_disagreement(responses)

        # Check if we should branch
        should_branch, reason = self.policy.should_branch(
            disagreement=disagreement,
            current_branches=len(self._graph.get_active_branches()),
        )

        if should_branch and reason:
            # Find the latest node to branch from
            latest_nodes = self._graph.get_leaf_nodes()
            if latest_nodes:
                from_node = latest_nodes[0]
                branch_name = f"branch_{len(self._graph.branches)}"

                self._graph.create_branch(
                    from_node_id=from_node.id,
                    reason=reason,
                    name=branch_name,
                    hypothesis=f"Alternative approach at round {round_num + 1}",
                )

                self._emit_event(
                    "branch_created",
                    {
                        "branch_name": branch_name,
                        "reason": reason.value,
                        "disagreement": disagreement,
                    },
                )

    def _calculate_disagreement(self, responses: dict[str, str]) -> float:
        """Calculate disagreement score between responses."""
        if len(responses) < 2:
            return 0.0

        # Simple disagreement detection based on opposing keywords
        disagreement_markers = [
            "disagree",
            "wrong",
            "incorrect",
            "no,",
            "however",
            "but",
            "on the contrary",
            "opposite",
            "instead",
        ]

        total_markers = 0
        for response in responses.values():
            response_lower = response.lower()
            for marker in disagreement_markers:
                if marker in response_lower:
                    total_markers += 1

        # Normalize to 0-1 scale
        max_possible = len(responses) * len(disagreement_markers)
        return min(1.0, total_markers / max(1, max_possible) * 5)  # Scale up

    async def _check_convergence_and_merge(
        self,
        run_agent_fn: Optional[Callable],
        context: Optional[dict],
    ) -> None:
        """Check for convergent branches and merge them."""
        if not self._graph:
            return

        convergent = self._graph.check_convergence()

        for branch_a, branch_b, score in convergent:
            if score >= self.policy.convergence_threshold:
                # Generate synthesis for merge
                synthesis = await self._generate_merge_synthesis(
                    [branch_a.id, branch_b.id],
                    run_agent_fn,
                    context,
                )

                # Perform merge
                self._graph.merge_branches(
                    branch_ids=[branch_a.id, branch_b.id],
                    strategy=self.policy.merge_strategy,
                    synthesizer_agent_id=self.agents[0].name if self.agents else "system",
                    synthesis_content=synthesis,
                )

                self._emit_event(
                    "branches_merged",
                    {
                        "branches": [branch_a.id, branch_b.id],
                        "strategy": self.policy.merge_strategy.value,
                        "convergence_score": score,
                    },
                )

    async def _generate_merge_synthesis(
        self,
        branch_ids: list[str],
        run_agent_fn: Optional[Callable],
        context: Optional[dict],
    ) -> str:
        """Generate synthesis content for merging branches."""
        if not self._graph or not self.agents:
            return "Branches merged."

        # Collect content from each branch
        branch_contents = []
        for bid in branch_ids:
            nodes = self._graph.get_branch_nodes(bid)
            if nodes:
                branch_contents.append(f"Branch {bid}: {nodes[-1].content[:200]}")

        prompt = (
            f"Synthesize the following viewpoints into a coherent conclusion:\n\n"
            f"{chr(10).join(branch_contents)}"
        )

        if run_agent_fn:
            return await run_agent_fn(self.agents[0], prompt, context)
        else:
            return await self.agents[0].generate(prompt, context)

    async def _generate_synthesis(
        self,
        task: str,
        run_agent_fn: Optional[Callable],
        context: Optional[dict],
    ) -> str:
        """Generate final synthesis of the debate."""
        if not self._graph or not self.agents:
            return ""

        history = self._get_debate_history()
        prompt = (
            f"Task: {task}\n\n"
            f"Debate history:\n{history}\n\n"
            f"Provide a final synthesis and conclusion."
        )

        if run_agent_fn:
            return await run_agent_fn(self.agents[0], prompt, context)
        else:
            return await self.agents[0].generate(prompt, context)

    def _build_result(
        self,
        debate_id: str,
        task: str,
        synthesis: str,
        total_rounds: int,
    ) -> GraphDebateResult:
        """Build the final debate result."""
        nodes: list[GraphNode] = []
        branches: list[GraphBranch] = []

        if self._graph:
            # Convert nodes
            for node in self._graph.nodes.values():
                round_num = node.metadata.get("round", 0)
                nodes.append(GraphNode.from_debate_node(node, round_num))

            # Convert branches
            for branch in self._graph.branches.values():
                branches.append(
                    GraphBranch(
                        id=branch.id,
                        name=branch.name,
                        start_node_id=branch.start_node_id,
                        hypothesis=branch.hypothesis,
                        is_active=branch.is_active,
                        is_merged=branch.is_merged,
                    )
                )

        return GraphDebateResult(
            id=debate_id,
            task=task,
            nodes=nodes,
            branches=branches,
            synthesis=synthesis,
            total_rounds=total_rounds,
        )

    def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit a debate event."""
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")

        logger.debug(f"graph_debate_event type={event_type} data={data}")

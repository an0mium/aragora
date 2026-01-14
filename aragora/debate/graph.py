"""
Debate Graph - Graph-based debates with counterfactual branching.

Enables non-linear debate structures where arguments can branch when agents
strongly disagree, explore alternative reasoning paths in parallel, and
merge when paths converge.

Key concepts:
- DebateNode: A single point in the debate with claims and responses
- DebateGraph: DAG structure allowing branches and merges
- BranchPolicy: Rules for when to create branches
- ConvergenceScorer: Detects when branches should merge
- GraphReplayBuilder: Reconstruct and replay graph debates
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)
import uuid


class NodeType(Enum):
    """Type of debate graph node."""

    ROOT = "root"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    BRANCH_POINT = "branch_point"
    MERGE_POINT = "merge_point"
    COUNTERFACTUAL = "counterfactual"
    CONCLUSION = "conclusion"


class BranchReason(Enum):
    """Why a branch was created."""

    HIGH_DISAGREEMENT = "high_disagreement"
    ALTERNATIVE_APPROACH = "alternative_approach"
    COUNTERFACTUAL_EXPLORATION = "counterfactual_exploration"
    RISK_MITIGATION = "risk_mitigation"
    UNCERTAINTY = "uncertainty"
    USER_REQUESTED = "user_requested"


class MergeStrategy(Enum):
    """How to merge branches back together."""

    BEST_PATH = "best_path"  # Pick the winning branch
    SYNTHESIS = "synthesis"  # Combine insights from all branches
    VOTE = "vote"  # Agents vote on best outcome
    WEIGHTED = "weighted"  # Weight by confidence scores
    PRESERVE_ALL = "preserve_all"  # Keep all as alternatives


@dataclass
class DebateNode:
    """A node in the debate graph."""

    id: str
    node_type: NodeType
    agent_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Graph structure
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    branch_id: Optional[str] = None

    # Metrics
    confidence: float = 0.0
    agreement_scores: dict[str, float] = field(default_factory=dict)

    # Metadata
    claims: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        """Compute content hash for verification."""
        data = f"{self.agent_id}:{self.content}:{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "agent_id": self.agent_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "branch_id": self.branch_id,
            "confidence": self.confidence,
            "agreement_scores": self.agreement_scores,
            "claims": self.claims,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "hash": self.hash(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DebateNode":
        """Deserialize node from dictionary."""
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            agent_id=data["agent_id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            branch_id=data.get("branch_id"),
            confidence=data.get("confidence", 0.0),
            agreement_scores=data.get("agreement_scores", {}),
            claims=data.get("claims", []),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Branch:
    """A branch in the debate graph."""

    id: str
    name: str
    reason: BranchReason
    start_node_id: str
    end_node_id: Optional[str] = None

    # Branch metadata
    hypothesis: str = ""
    confidence: float = 0.0
    is_active: bool = True
    is_merged: bool = False
    merged_into: Optional[str] = None

    # Metrics
    node_count: int = 0
    total_agreement: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "reason": self.reason.value,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "is_active": self.is_active,
            "is_merged": self.is_merged,
            "merged_into": self.merged_into,
            "node_count": self.node_count,
            "total_agreement": self.total_agreement,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Branch":
        return cls(
            id=data["id"],
            name=data["name"],
            reason=BranchReason(data["reason"]),
            start_node_id=data["start_node_id"],
            end_node_id=data.get("end_node_id"),
            hypothesis=data.get("hypothesis", ""),
            confidence=data.get("confidence", 0.0),
            is_active=data.get("is_active", True),
            is_merged=data.get("is_merged", False),
            merged_into=data.get("merged_into"),
            node_count=data.get("node_count", 0),
            total_agreement=data.get("total_agreement", 0.0),
        )


@dataclass
class MergeResult:
    """Result of merging branches."""

    merged_node_id: str
    source_branch_ids: list[str]
    strategy: MergeStrategy
    synthesis: str
    confidence: float
    insights_preserved: list[str] = field(default_factory=list)
    conflicts_resolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize merge result to dictionary."""
        return {
            "merged_node_id": self.merged_node_id,
            "source_branch_ids": self.source_branch_ids,
            "strategy": self.strategy.value,
            "synthesis": self.synthesis,
            "confidence": self.confidence,
            "insights_preserved": self.insights_preserved,
            "conflicts_resolved": self.conflicts_resolved,
        }


@dataclass
class BranchPolicy:
    """Policy for when to create branches."""

    # Thresholds
    disagreement_threshold: float = 0.6  # Branch if disagreement > this
    uncertainty_threshold: float = 0.7  # Branch if uncertainty > this
    min_alternative_score: float = 0.3  # Min score for alternative to branch

    # Limits
    max_branches: int = 4
    max_depth: int = 5

    # Conditions
    allow_counterfactuals: bool = True
    allow_user_branches: bool = True
    auto_merge_on_convergence: bool = True
    convergence_threshold: float = 0.8

    def should_branch(
        self,
        disagreement: float,
        uncertainty: float,
        current_branches: int,
        current_depth: int,
        alternative_score: float = 0.0,
    ) -> tuple[bool, Optional[BranchReason]]:
        """Determine if a branch should be created."""

        # Check limits
        if current_branches >= self.max_branches:
            return False, None
        if current_depth >= self.max_depth:
            return False, None

        # Check conditions
        if disagreement > self.disagreement_threshold:
            return True, BranchReason.HIGH_DISAGREEMENT

        if uncertainty > self.uncertainty_threshold:
            return True, BranchReason.UNCERTAINTY

        if alternative_score > self.min_alternative_score:
            return True, BranchReason.ALTERNATIVE_APPROACH

        return False, None


class ConvergenceScorer:
    """Detects when branches should merge based on convergence."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def score_convergence(
        self,
        branch_a: Branch,
        branch_b: Branch,
        nodes_a: list[DebateNode],
        nodes_b: list[DebateNode],
    ) -> float:
        """Score how much two branches have converged (0-1)."""

        if not nodes_a or not nodes_b:
            return 0.0

        # Get latest claims from each branch
        claims_a = set()
        claims_b = set()

        for node in nodes_a[-3:]:  # Last 3 nodes
            claims_a.update(node.claims)

        for node in nodes_b[-3:]:
            claims_b.update(node.claims)

        if not claims_a and not claims_b:
            # No claims - check content similarity
            return self._content_similarity(nodes_a[-1].content, nodes_b[-1].content)

        # Jaccard similarity of claims
        if claims_a or claims_b:
            intersection = len(claims_a & claims_b)
            union = len(claims_a | claims_b)
            claim_similarity = intersection / union if union > 0 else 0.0
        else:
            claim_similarity = 0.0

        # Average confidence convergence
        conf_a = sum(n.confidence for n in nodes_a[-3:]) / min(3, len(nodes_a))
        conf_b = sum(n.confidence for n in nodes_b[-3:]) / min(3, len(nodes_b))
        conf_similarity = 1.0 - abs(conf_a - conf_b)

        # Combined score
        return 0.7 * claim_similarity + 0.3 * conf_similarity

    def _content_similarity(self, content_a: str, content_b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(content_a.lower().split())
        words_b = set(content_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

    def should_merge(
        self,
        branch_a: Branch,
        branch_b: Branch,
        nodes_a: list[DebateNode],
        nodes_b: list[DebateNode],
    ) -> bool:
        """Check if two branches should be merged."""
        score = self.score_convergence(branch_a, branch_b, nodes_a, nodes_b)
        return score >= self.threshold


class DebateGraph:
    """
    Graph-based debate structure with branching and merging.

    Enables counterfactual exploration where agents can pursue
    alternative reasoning paths and later merge insights.
    """

    def __init__(
        self,
        debate_id: Optional[str] = None,
        branch_policy: Optional[BranchPolicy] = None,
    ):
        self.debate_id = debate_id or str(uuid.uuid4())
        self.policy = branch_policy or BranchPolicy()
        self.convergence_scorer = ConvergenceScorer(self.policy.convergence_threshold)

        # Graph structure
        self.nodes: dict[str, DebateNode] = {}
        self.branches: dict[str, Branch] = {}
        self.root_id: Optional[str] = None

        # Main branch (trunk)
        self.main_branch_id = "main"
        self.branches[self.main_branch_id] = Branch(
            id=self.main_branch_id,
            name="Main",
            reason=BranchReason.USER_REQUESTED,
            start_node_id="",
        )

        # Tracking
        self.merge_history: list[MergeResult] = []
        self.created_at = datetime.now()

        # Caching for graph traversal operations
        self._cache_version: int = 0
        self._branch_nodes_cache: dict[tuple[str, int], list[DebateNode]] = {}
        self._path_cache: dict[tuple[str, int], list[DebateNode]] = {}
        self._leaf_nodes_cache: tuple[int, list[DebateNode]] | None = None
        self._active_branches_cache: tuple[int, list[Branch]] | None = None

    def _invalidate_cache(self) -> None:
        """Invalidate all caches when graph structure changes."""
        self._cache_version += 1
        # Clear caches to prevent memory accumulation from orphaned entries
        self._branch_nodes_cache.clear()
        self._path_cache.clear()
        self._leaf_nodes_cache = None
        self._active_branches_cache = None

    def add_node(
        self,
        node_type: NodeType,
        agent_id: str,
        content: str,
        parent_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        claims: Optional[list[str]] = None,
        evidence: Optional[list[str]] = None,
        confidence: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> DebateNode:
        """Add a node to the debate graph."""

        node_id = str(uuid.uuid4())[:8]
        branch_id = branch_id or self.main_branch_id

        node = DebateNode(
            id=node_id,
            node_type=node_type,
            agent_id=agent_id,
            content=content,
            parent_ids=[parent_id] if parent_id else [],
            branch_id=branch_id,
            claims=claims or [],
            evidence=evidence or [],
            confidence=confidence,
            metadata=metadata or {},
        )

        self.nodes[node_id] = node

        # Update parent's children
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].child_ids.append(node_id)

        # Set root if first node
        if self.root_id is None:
            self.root_id = node_id
            self.branches[self.main_branch_id].start_node_id = node_id

        # Update branch stats
        if branch_id in self.branches:
            self.branches[branch_id].node_count += 1
            self.branches[branch_id].end_node_id = node_id

        # Invalidate caches when graph changes
        self._invalidate_cache()

        return node

    def create_branch(
        self,
        from_node_id: str,
        reason: BranchReason,
        name: str,
        hypothesis: str = "",
    ) -> Branch:
        """Create a new branch from a node."""

        if from_node_id not in self.nodes:
            raise ValueError(f"Node {from_node_id} not found")

        branch_id = str(uuid.uuid4())[:8]
        branch = Branch(
            id=branch_id,
            name=name,
            reason=reason,
            start_node_id=from_node_id,
            hypothesis=hypothesis,
        )

        self.branches[branch_id] = branch

        # Mark the source node as a branch point
        source_node = self.nodes[from_node_id]
        if source_node.node_type != NodeType.BRANCH_POINT:
            source_node.metadata["is_branch_point"] = True
            source_node.metadata["branch_ids"] = source_node.metadata.get("branch_ids", [])
            source_node.metadata["branch_ids"].append(branch_id)

        # Invalidate caches when graph changes
        self._invalidate_cache()

        return branch

    def merge_branches(
        self,
        branch_ids: list[str],
        strategy: MergeStrategy,
        synthesizer_agent_id: str,
        synthesis_content: str,
    ) -> MergeResult:
        """Merge multiple branches into a single node."""

        # Validate branches
        for bid in branch_ids:
            if bid not in self.branches:
                raise ValueError(f"Branch {bid} not found")

        # Get end nodes of each branch
        parent_ids = []
        for bid in branch_ids:
            branch = self.branches[bid]
            if branch.end_node_id:
                parent_ids.append(branch.end_node_id)

        # Create merge node
        merge_node = self.add_node(
            node_type=NodeType.MERGE_POINT,
            agent_id=synthesizer_agent_id,
            content=synthesis_content,
            branch_id=self.main_branch_id,
            metadata={"merged_from": branch_ids, "strategy": strategy.value},
        )

        # Set multiple parents
        merge_node.parent_ids = parent_ids

        # Update parent nodes' children
        for pid in parent_ids:
            if pid in self.nodes:
                self.nodes[pid].child_ids.append(merge_node.id)

        # Mark branches as merged
        for bid in branch_ids:
            self.branches[bid].is_active = False
            self.branches[bid].is_merged = True
            self.branches[bid].merged_into = merge_node.id

        # Collect insights from each branch
        insights = []
        for bid in branch_ids:
            branch_nodes = self.get_branch_nodes(bid)
            for node in branch_nodes:
                insights.extend(node.claims)

        result = MergeResult(
            merged_node_id=merge_node.id,
            source_branch_ids=branch_ids,
            strategy=strategy,
            synthesis=synthesis_content,
            confidence=merge_node.confidence,
            insights_preserved=list(set(insights)),
        )

        self.merge_history.append(result)

        # Invalidate caches when graph changes
        self._invalidate_cache()

        return result

    def get_branch_nodes(self, branch_id: str) -> list[DebateNode]:
        """Get all nodes in a branch (cached)."""
        cache_key = (branch_id, self._cache_version)
        if cache_key in self._branch_nodes_cache:
            return self._branch_nodes_cache[cache_key]

        result = [n for n in self.nodes.values() if n.branch_id == branch_id]
        self._branch_nodes_cache[cache_key] = result
        return result

    def get_active_branches(self) -> list[Branch]:
        """Get all active (unmerged) branches (cached)."""
        if self._active_branches_cache and self._active_branches_cache[0] == self._cache_version:
            return self._active_branches_cache[1]

        result = [b for b in self.branches.values() if b.is_active]
        self._active_branches_cache = (self._cache_version, result)
        return result

    def get_path_to_node(self, node_id: str) -> list[DebateNode]:
        """Get the path from root to a specific node (cached)."""
        if node_id not in self.nodes:
            return []

        cache_key = (node_id, self._cache_version)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        path = []
        current_id: str | None = node_id
        visited: set[str] = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            node = self.nodes.get(current_id)
            if node:
                path.append(node)
                current_id = node.parent_ids[0] if node.parent_ids else None
            else:
                break

        result = list(reversed(path))
        self._path_cache[cache_key] = result
        return result

    def get_leaf_nodes(self) -> list[DebateNode]:
        """Get all nodes with no children (current endpoints) (cached)."""
        if self._leaf_nodes_cache and self._leaf_nodes_cache[0] == self._cache_version:
            return self._leaf_nodes_cache[1]

        result = [n for n in self.nodes.values() if not n.child_ids]
        self._leaf_nodes_cache = (self._cache_version, result)
        return result

    def check_convergence(self) -> list[tuple[Branch, Branch, float]]:
        """Check for convergent branches that could be merged."""
        candidates = []
        active = self.get_active_branches()

        for i, branch_a in enumerate(active):
            for branch_b in active[i + 1 :]:
                nodes_a = self.get_branch_nodes(branch_a.id)
                nodes_b = self.get_branch_nodes(branch_b.id)

                score = self.convergence_scorer.score_convergence(
                    branch_a, branch_b, nodes_a, nodes_b
                )

                if score >= self.policy.convergence_threshold:
                    candidates.append((branch_a, branch_b, score))

        return sorted(candidates, key=lambda x: -x[2])

    def to_dict(self) -> dict:
        """Serialize graph to dictionary."""
        return {
            "debate_id": self.debate_id,
            "root_id": self.root_id,
            "main_branch_id": self.main_branch_id,
            "created_at": self.created_at.isoformat(),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "branches": {bid: b.to_dict() for bid, b in self.branches.items()},
            "merge_history": [
                {
                    "merged_node_id": m.merged_node_id,
                    "source_branch_ids": m.source_branch_ids,
                    "strategy": m.strategy.value,
                    "synthesis": m.synthesis,
                    "confidence": m.confidence,
                    "insights_preserved": m.insights_preserved,
                }
                for m in self.merge_history
            ],
            "policy": {
                "disagreement_threshold": self.policy.disagreement_threshold,
                "uncertainty_threshold": self.policy.uncertainty_threshold,
                "max_branches": self.policy.max_branches,
                "max_depth": self.policy.max_depth,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DebateGraph":
        """Deserialize graph from dictionary."""
        policy = BranchPolicy(
            disagreement_threshold=data["policy"]["disagreement_threshold"],
            uncertainty_threshold=data["policy"]["uncertainty_threshold"],
            max_branches=data["policy"]["max_branches"],
            max_depth=data["policy"]["max_depth"],
        )

        graph = cls(debate_id=data["debate_id"], branch_policy=policy)
        graph.root_id = data["root_id"]
        graph.main_branch_id = data["main_branch_id"]
        graph.created_at = datetime.fromisoformat(data["created_at"])

        # Load nodes
        graph.nodes = {nid: DebateNode.from_dict(ndata) for nid, ndata in data["nodes"].items()}

        # Load branches
        graph.branches = {bid: Branch.from_dict(bdata) for bid, bdata in data["branches"].items()}

        return graph


class GraphReplayBuilder:
    """Replay and analyze graph-based debates."""

    def __init__(self, graph: DebateGraph):
        self.graph = graph

    def replay_branch(
        self,
        branch_id: str,
        callback: Optional[Callable[[DebateNode, int], None]] = None,
    ) -> list[DebateNode]:
        """Replay a specific branch in order."""
        nodes = self.graph.get_branch_nodes(branch_id)

        # Sort by timestamp
        nodes.sort(key=lambda n: n.timestamp)

        if callback:
            for i, node in enumerate(nodes):
                callback(node, i)

        return nodes

    def replay_full(
        self,
        callback: Optional[Callable[[DebateNode, str, int], None]] = None,
    ) -> dict[str, list[DebateNode]]:
        """Replay entire graph, branch by branch."""
        result = {}

        for branch_id in self.graph.branches:
            nodes = self.replay_branch(branch_id)
            result[branch_id] = nodes

            if callback:
                for i, node in enumerate(nodes):
                    callback(node, branch_id, i)

        return result

    def get_counterfactual_paths(self) -> list[list[DebateNode]]:
        """Get all counterfactual exploration paths."""
        paths = []

        for branch in self.graph.branches.values():
            if branch.reason == BranchReason.COUNTERFACTUAL_EXPLORATION:
                path = []
                nodes = self.graph.get_branch_nodes(branch.id)

                # Include path from root to branch start
                if nodes:
                    prefix = self.graph.get_path_to_node(branch.start_node_id)
                    path.extend(prefix)
                    path.extend(nodes)

                if path:
                    paths.append(path)

        return paths

    def generate_summary(self) -> dict:
        """Generate a summary of the debate graph."""
        return {
            "debate_id": self.graph.debate_id,
            "total_nodes": len(self.graph.nodes),
            "total_branches": len(self.graph.branches),
            "active_branches": len(self.graph.get_active_branches()),
            "merges": len(self.graph.merge_history),
            "branch_reasons": {
                reason.value: sum(1 for b in self.graph.branches.values() if b.reason == reason)
                for reason in BranchReason
            },
            "agents": list(set(n.agent_id for n in self.graph.nodes.values())),
            "leaf_nodes": len(self.graph.get_leaf_nodes()),
        }


class GraphDebateOrchestrator:
    """Orchestrate graph-based debates with automatic branching."""

    def __init__(
        self,
        agents: list,  # List of Agent objects
        policy: Optional[BranchPolicy] = None,
    ):
        self.agents = agents
        self.policy = policy or BranchPolicy()
        self.graph = DebateGraph(branch_policy=self.policy)

    def _emit_graph_event(self, event_emitter, event_type: str, data: dict, debate_id: str):
        """Emit a graph debate event via WebSocket."""
        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            type_map = {
                "node": StreamEventType.GRAPH_NODE_ADDED,
                "branch": StreamEventType.GRAPH_BRANCH_CREATED,
                "merge": StreamEventType.GRAPH_BRANCH_MERGED,
            }
            stream_type = type_map.get(event_type)
            if stream_type and event_emitter:
                event_emitter.emit(
                    StreamEvent(  # type: ignore[call-arg]
                        type=stream_type,
                        data=data,
                        debate_id=debate_id,
                    )
                )
        except Exception as e:
            logger.debug(f"Stream event emission failed (non-critical): {e}")

    async def run_debate(
        self,
        task: str,
        max_rounds: int = 5,
        run_agent_fn: Optional[Callable] = None,
        on_node: Optional[Callable[[DebateNode], None]] = None,
        on_branch: Optional[Callable[[Branch], None]] = None,
        on_merge: Optional[Callable[[MergeResult], None]] = None,
        event_emitter=None,
        debate_id: str = "",
    ) -> DebateGraph:
        """
        Run a graph-based debate with automatic branching.

        Args:
            task: The debate topic/question
            max_rounds: Maximum rounds per branch
            run_agent_fn: Async function(agent, prompt, context) -> str
            on_node: Callback for new nodes
            on_branch: Callback for new branches
            on_merge: Callback for merges
            event_emitter: Optional WebSocket event emitter for real-time streaming
            debate_id: Debate ID for event scoping

        Flow:
        1. Start with a root proposal
        2. Have agents critique and respond
        3. Detect disagreements and create branches
        4. Explore counterfactuals in parallel
        5. Detect convergence and merge branches
        6. Produce a final synthesis
        """

        # Create event-emitting wrapper callbacks
        def emit_node(node: DebateNode) -> None:
            if on_node:
                on_node(node)
            if event_emitter:
                self._emit_graph_event(event_emitter, "node", node.to_dict(), debate_id)

        def emit_branch(branch: Branch) -> None:
            if on_branch:
                on_branch(branch)
            if event_emitter:
                self._emit_graph_event(event_emitter, "branch", branch.to_dict(), debate_id)

        def emit_merge(merge_result: MergeResult) -> None:
            if on_merge:
                on_merge(merge_result)
            if event_emitter:
                self._emit_graph_event(event_emitter, "merge", merge_result.to_dict(), debate_id)

        # Create root node
        root = self.graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content=task,
            confidence=1.0,
        )

        # Emit root node event
        emit_node(root)

        # If no run function provided, return initialized graph
        if run_agent_fn is None:
            return self.graph

        # Track active branch heads (branch_id -> current_node_id)
        branch_heads: dict[str, str] = {self.graph.main_branch_id: root.id}

        for round_num in range(max_rounds):
            # Process each active branch
            active_branches = self.graph.get_active_branches()

            for branch in active_branches:
                if branch.id not in branch_heads:
                    continue

                current_node_id = branch_heads[branch.id]
                current_node = self.graph.nodes[current_node_id]

                # Build context from branch history
                branch_nodes = self.graph.get_branch_nodes(branch.id)
                context = self._build_context(branch_nodes)

                # Collect responses from all agents
                responses: list[tuple[str, str, float]] = []

                for agent in self.agents:
                    prompt = self._build_prompt(
                        task=task,
                        round_num=round_num,
                        current_content=current_node.content,
                        branch_hypothesis=branch.hypothesis,
                    )

                    try:
                        response = await run_agent_fn(agent, prompt, context)
                        confidence = self._extract_confidence(response)
                        responses.append((agent.name, response, confidence))
                    except Exception as e:
                        responses.append((agent.name, f"Error: {e}", 0.0))

                # Evaluate disagreement
                disagreement, alternative = self.evaluate_disagreement(responses)

                # Check if we should branch
                should_branch, branch_reason = self.policy.should_branch(
                    disagreement=disagreement,
                    uncertainty=1.0 - max(r[2] for r in responses) if responses else 1.0,
                    current_branches=len(active_branches),
                    current_depth=round_num,
                    alternative_score=0.5 if alternative else 0.0,
                )

                if should_branch and branch_reason and alternative:
                    # Create a new branch for the divergent view
                    new_branch = self.graph.create_branch(
                        from_node_id=current_node_id,
                        reason=branch_reason,
                        name=f"Branch-{len(self.graph.branches)}",
                        hypothesis=alternative[:200] if alternative else "",
                    )

                    emit_branch(new_branch)

                    # Add first node to new branch
                    divergent_agent = next(
                        (r[0] for r in responses if r[1] == alternative),
                        responses[0][0] if responses else "unknown",
                    )
                    branch_node = self.graph.add_node(
                        node_type=NodeType.COUNTERFACTUAL,
                        agent_id=divergent_agent,
                        content=alternative,
                        parent_id=current_node_id,
                        branch_id=new_branch.id,
                        confidence=self._extract_confidence(alternative),
                    )
                    branch_heads[new_branch.id] = branch_node.id

                    emit_node(branch_node)

                # Add main response to current branch
                if responses:
                    # Pick highest confidence response for main branch
                    best_response = max(responses, key=lambda r: r[2])
                    node_type = NodeType.PROPOSAL if round_num == 0 else NodeType.SYNTHESIS

                    new_node = self.graph.add_node(
                        node_type=node_type,
                        agent_id=best_response[0],
                        content=best_response[1],
                        parent_id=current_node_id,
                        branch_id=branch.id,
                        confidence=best_response[2],
                        claims=self._extract_claims(best_response[1]),
                    )
                    branch_heads[branch.id] = new_node.id

                    emit_node(new_node)

            # Check for convergence and merge
            convergent_pairs = self.graph.check_convergence()
            for branch_a, branch_b, score in convergent_pairs:
                if branch_a.is_active and branch_b.is_active:
                    # Synthesize the merge
                    nodes_a = self.graph.get_branch_nodes(branch_a.id)
                    nodes_b = self.graph.get_branch_nodes(branch_b.id)

                    synthesis = self._synthesize_branches(nodes_a, nodes_b)

                    merge_result = self.graph.merge_branches(
                        branch_ids=[branch_a.id, branch_b.id],
                        strategy=MergeStrategy.SYNTHESIS,
                        synthesizer_agent_id="system",
                        synthesis_content=synthesis,
                    )

                    # Update branch heads
                    if branch_a.id in branch_heads:
                        del branch_heads[branch_a.id]
                    if branch_b.id in branch_heads:
                        del branch_heads[branch_b.id]
                    branch_heads[self.graph.main_branch_id] = merge_result.merged_node_id

                    emit_merge(merge_result)

            # Check if all branches have converged
            if len(self.graph.get_active_branches()) <= 1:
                break

        # Create final conclusion node
        leaf_nodes = self.graph.get_leaf_nodes()
        if leaf_nodes:
            final_content = self._create_final_synthesis(leaf_nodes)
            self.graph.add_node(
                node_type=NodeType.CONCLUSION,
                agent_id="system",
                content=final_content,
                parent_id=leaf_nodes[0].id if leaf_nodes else None,
                branch_id=self.graph.main_branch_id,
                confidence=max(n.confidence for n in leaf_nodes) if leaf_nodes else 0.5,
            )

        return self.graph

    def _build_context(self, nodes: list[DebateNode]) -> str:
        """Build context string from branch nodes."""
        if not nodes:
            return ""

        context_parts = []
        for node in nodes[-5:]:  # Last 5 nodes
            context_parts.append(f"[{node.agent_id}]: {node.content[:500]}")

        return "\n\n".join(context_parts)

    def _build_prompt(
        self,
        task: str,
        round_num: int,
        current_content: str,
        branch_hypothesis: str = "",
    ) -> str:
        """Build prompt for agent response."""
        prompt_parts = [f"Task: {task}"]

        if round_num == 0:
            prompt_parts.append("Provide your initial analysis and position on this topic.")
        else:
            prompt_parts.append(f"Previous response: {current_content[:500]}")
            prompt_parts.append(
                "Critique or build upon this response. State your confidence (0-100%)."
            )

        if branch_hypothesis:
            prompt_parts.append(f"Explore this alternative view: {branch_hypothesis}")

        return "\n\n".join(prompt_parts)

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response."""
        import re

        # Look for explicit confidence
        match = re.search(r"confidence[:\s]+(\d+)%?", response.lower())
        if match:
            return min(1.0, int(match.group(1)) / 100.0)

        # Look for percentage
        match = re.search(r"(\d+)%\s*(?:confident|certain|sure)", response.lower())
        if match:
            return min(1.0, int(match.group(1)) / 100.0)

        # Default based on tone
        high_conf_words = ["certain", "definitely", "clearly", "obviously", "undoubtedly"]
        low_conf_words = ["perhaps", "maybe", "might", "possibly", "uncertain"]

        response_lower = response.lower()
        if any(w in response_lower for w in high_conf_words):
            return 0.8
        if any(w in response_lower for w in low_conf_words):
            return 0.4

        return 0.6  # Default moderate confidence

    def _extract_claims(self, response: str) -> list[str]:
        """Extract key claims from response."""
        import re

        claims = []

        # Look for numbered claims
        numbered = re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+?)(?:\n|$)", response)
        claims.extend(numbered[:5])

        # Look for bullet points
        bullets = re.findall(r"(?:^|\n)\s*[-*]\s*(.+?)(?:\n|$)", response)
        claims.extend(bullets[:5])

        # If no structured claims, take first sentence
        if not claims:
            sentences = re.split(r"[.!?]+", response)
            if sentences:
                claims.append(sentences[0].strip()[:200])

        return claims[:5]  # Limit to 5 claims

    def _synthesize_branches(
        self,
        nodes_a: list[DebateNode],
        nodes_b: list[DebateNode],
    ) -> str:
        """Create synthesis content from merging branches."""
        claims_a = set()
        claims_b = set()

        for node in nodes_a:
            claims_a.update(node.claims)
        for node in nodes_b:
            claims_b.update(node.claims)

        common = claims_a & claims_b
        unique_a = claims_a - claims_b
        unique_b = claims_b - claims_a

        synthesis_parts = ["## Branch Synthesis"]

        if common:
            synthesis_parts.append(f"**Agreed upon:** {', '.join(list(common)[:3])}")

        if unique_a:
            synthesis_parts.append(f"**From Branch A:** {', '.join(list(unique_a)[:2])}")

        if unique_b:
            synthesis_parts.append(f"**From Branch B:** {', '.join(list(unique_b)[:2])}")

        if nodes_a:
            synthesis_parts.append(f"**Final position from A:** {nodes_a[-1].content[:200]}")
        if nodes_b:
            synthesis_parts.append(f"**Final position from B:** {nodes_b[-1].content[:200]}")

        return "\n\n".join(synthesis_parts)

    def _create_final_synthesis(self, leaf_nodes: list[DebateNode]) -> str:
        """Create final conclusion from all leaf nodes."""
        if not leaf_nodes:
            return "No conclusion reached."

        if len(leaf_nodes) == 1:
            return f"**Conclusion:** {leaf_nodes[0].content}"

        # Multiple endpoints - summarize all
        parts = ["## Final Synthesis", ""]
        for i, node in enumerate(leaf_nodes[:5], 1):
            conf_str = f"({node.confidence:.0%} confidence)" if node.confidence else ""
            parts.append(f"**Path {i}** {conf_str}: {node.content[:300]}")

        # Find common claims
        all_claims = set()
        for node in leaf_nodes:
            all_claims.update(node.claims)

        if all_claims:
            parts.append(f"\n**Key claims across paths:** {', '.join(list(all_claims)[:5])}")

        return "\n\n".join(parts)

    def evaluate_disagreement(
        self,
        responses: list[tuple[str, str, float]],  # (agent_id, content, confidence)
    ) -> tuple[float, Optional[str]]:
        """
        Evaluate disagreement among agent responses.
        Returns (disagreement_score, alternative_content if branch needed)
        """
        if len(responses) < 2:
            return 0.0, None

        # Simple variance-based disagreement
        confidences = [r[2] for r in responses]
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)

        # Normalize to 0-1
        disagreement = min(1.0, variance * 4)

        # Find most divergent response
        if disagreement > self.policy.disagreement_threshold and responses:
            sorted_by_conf = sorted(responses, key=lambda x: x[2])
            return disagreement, sorted_by_conf[0][1]

        return disagreement, None

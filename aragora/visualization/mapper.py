"""
Argument Cartographer - builds directed graphs of debate logic in real-time.

This is a pure observer that reads debate events and constructs a graph
representation. It never modifies debate state or agent prompts.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import hashlib
import logging
import time

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of argument nodes in the debate graph."""
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    EVIDENCE = "evidence"
    CONCESSION = "concession"
    REBUTTAL = "rebuttal"
    VOTE = "vote"
    CONSENSUS = "consensus"


class EdgeRelation(Enum):
    """Types of logical relationships between arguments."""
    SUPPORTS = "supports"
    REFUTES = "refutes"
    MODIFIES = "modifies"
    RESPONDS_TO = "responds_to"
    CONCEDES_TO = "concedes_to"


@dataclass
class ArgumentNode:
    """A node in the argument graph representing a discrete claim or action."""
    id: str
    agent: str
    node_type: NodeType
    summary: str  # First 100 chars or extracted claim
    round_num: int
    timestamp: float
    full_content: Optional[str] = None  # Store full text for detailed views
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "agent": self.agent,
            "node_type": self.node_type.value,
            "summary": self.summary,
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ArgumentEdge:
    """An edge representing a logical relationship between arguments."""
    source_id: str
    target_id: str
    relation: EdgeRelation
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class ArgumentCartographer:
    """
    Builds a directed graph of debate logic in real-time.
    
    This is a pure observer - it reads events and builds a graph,
    but never modifies debate state, prompts, or other core systems.
    """
    nodes: Dict[str, ArgumentNode] = field(default_factory=dict)
    edges: List[ArgumentEdge] = field(default_factory=list)
    debate_id: Optional[str] = None
    topic: Optional[str] = None
    
    # Internal tracking for graph construction
    _last_proposal_id: Optional[str] = None
    _agent_last_node: Dict[str, str] = field(default_factory=dict)
    _round_proposals: Dict[int, str] = field(default_factory=dict)

    def set_debate_context(self, debate_id: str, topic: str) -> None:
        """Set the debate context for this cartographer instance."""
        self.debate_id = debate_id
        self.topic = topic

    def update_from_message(
        self, 
        agent: str, 
        content: str, 
        role: str, 
        round_num: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a debate message and update the graph.
        
        Returns the node ID of the created node.
        """
        node_id = self._make_id(agent, round_num, content)
        node_type = self._infer_type(role, content)
        
        summary = content[:100] + "..." if len(content) > 100 else content
        # Clean summary for display
        summary = summary.replace("\n", " ").strip()
        
        node = ArgumentNode(
            id=node_id,
            agent=agent,
            node_type=node_type,
            summary=summary,
            round_num=round_num,
            timestamp=time.time(),
            full_content=content,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        
        # Build graph edges based on node type and context
        self._link_node(node, agent, round_num)
        
        return node_id

    def update_from_critique(
        self, 
        critic_agent: str, 
        target_agent: str, 
        severity: float, 
        round_num: int,
        critique_text: Optional[str] = None
    ) -> Optional[str]:
        """
        Record a critique relationship between agents.
        
        Returns the edge ID if created, None otherwise.
        """
        critic_node_id = self._agent_last_node.get(critic_agent)
        target_node_id = self._agent_last_node.get(target_agent)
        
        if not critic_node_id or not target_node_id:
            return None
        
        # Determine relationship type based on severity
        if severity > 0.7:
            relation = EdgeRelation.REFUTES
        elif severity > 0.3:
            relation = EdgeRelation.MODIFIES
        else:
            relation = EdgeRelation.RESPONDS_TO
        
        edge = ArgumentEdge(
            source_id=critic_node_id,
            target_id=target_node_id,
            relation=relation,
            weight=severity,
            metadata={"critique_text": critique_text} if critique_text else {},
        )
        self.edges.append(edge)
        
        return f"{critic_node_id}->{target_node_id}"

    def update_from_vote(
        self, 
        agent: str, 
        vote_value: str, 
        round_num: int
    ) -> str:
        """Record a vote as a node in the graph."""
        node_id = self._make_id(agent, round_num, f"vote:{vote_value}")
        
        node = ArgumentNode(
            id=node_id,
            agent=agent,
            node_type=NodeType.VOTE,
            summary=f"Votes: {vote_value}",
            round_num=round_num,
            timestamp=time.time(),
            metadata={"vote_value": vote_value},
        )
        self.nodes[node_id] = node
        
        # Link vote to the round's proposal
        if round_num in self._round_proposals:
            self.edges.append(ArgumentEdge(
                source_id=node_id,
                target_id=self._round_proposals[round_num],
                relation=EdgeRelation.RESPONDS_TO,
            ))
        
        return node_id

    def update_from_consensus(
        self, 
        result: str, 
        round_num: int,
        vote_counts: Optional[Dict[str, int]] = None
    ) -> str:
        """Record the consensus outcome."""
        node_id = f"consensus_{round_num}"
        
        node = ArgumentNode(
            id=node_id,
            agent="system",
            node_type=NodeType.CONSENSUS,
            summary=f"Consensus: {result}",
            round_num=round_num,
            timestamp=time.time(),
            metadata={"result": result, "vote_counts": vote_counts or {}},
        )
        self.nodes[node_id] = node
        
        # Link consensus to all votes in this round
        for nid, n in self.nodes.items():
            if n.node_type == NodeType.VOTE and n.round_num == round_num:
                self.edges.append(ArgumentEdge(
                    source_id=nid,
                    target_id=node_id,
                    relation=EdgeRelation.SUPPORTS,
                ))
        
        return node_id

    def export_mermaid(self, direction: str = "TD") -> str:
        """
        Generate Mermaid.js diagram code.
        
        Args:
            direction: Graph direction - TD (top-down), LR (left-right)
        
        Returns:
            Mermaid.js diagram as a string.
        """
        lines = [f"graph {direction}"]
        
        # Style definitions for different node types
        lines.append("    %% Node type styles")
        lines.append("    classDef proposal fill:#4CAF50,stroke:#2E7D32,color:#fff")
        lines.append("    classDef critique fill:#FF5722,stroke:#D84315,color:#fff")
        lines.append("    classDef evidence fill:#9C27B0,stroke:#6A1B9A,color:#fff")
        lines.append("    classDef concession fill:#FF9800,stroke:#E65100,color:#fff")
        lines.append("    classDef rebuttal fill:#F44336,stroke:#C62828,color:#fff")
        lines.append("    classDef vote fill:#607D8B,stroke:#37474F,color:#fff")
        lines.append("    classDef consensus fill:#2196F3,stroke:#1565C0,color:#fff")
        lines.append("")
        
        # Group nodes by round for subgraphs
        rounds: Dict[int, List[str]] = {}
        for nid, node in self.nodes.items():
            rounds.setdefault(node.round_num, []).append(nid)
        
        # Generate nodes within round subgraphs
        for round_num in sorted(rounds.keys()):
            lines.append(f"    subgraph Round_{round_num}[Round {round_num}]")
            for nid in rounds[round_num]:
                node = self.nodes[nid]
                safe_summary = self._sanitize_for_mermaid(node.summary)[:50]
                label = f"{node.agent}: {safe_summary}"
                lines.append(f'        {nid}["{label}"]')
            lines.append("    end")
            lines.append("")
        
        # Apply styles to nodes
        lines.append("    %% Apply node styles")
        for nid, node in self.nodes.items():
            lines.append(f"    class {nid} {node.node_type.value}")
        lines.append("")
        
        # Generate edges with relationship labels
        lines.append("    %% Edges")
        for edge in self.edges:
            if edge.source_id in self.nodes and edge.target_id in self.nodes:
                arrow = self._get_mermaid_arrow(edge.relation)
                lines.append(f"    {edge.source_id} {arrow} {edge.target_id}")
        
        return "\n".join(lines)

    def export_json(self, include_full_content: bool = False) -> str:
        """
        Export graph as JSON for downstream analysis.
        
        Args:
            include_full_content: Whether to include full message content.
        """
        nodes_data = []
        for node in self.nodes.values():
            node_dict = node.to_dict()
            if include_full_content:
                node_dict["full_content"] = node.full_content
            nodes_data.append(node_dict)
        
        return json.dumps({
            "debate_id": self.debate_id,
            "topic": self.topic,
            "nodes": nodes_data,
            "edges": [e.to_dict() for e in self.edges],
            "metadata": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "exported_at": time.time(),
            }
        }, indent=2, default=str)

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the argument graph.

        Returns stats matching the frontend GraphStats interface:
        - node_count, edge_count: Basic counts
        - max_depth: Maximum chain length from root
        - avg_branching: Average outgoing edges per node
        - complexity_score: 0-1 normalized complexity metric
        - claim_count, rebuttal_count: Type-specific counts
        """
        node_types: dict[str, int] = {}
        for node in self.nodes.values():
            node_types[node.node_type.value] = node_types.get(node.node_type.value, 0) + 1

        edge_types: dict[str, int] = {}
        for edge in self.edges:
            edge_types[edge.relation.value] = edge_types.get(edge.relation.value, 0) + 1

        agents = set(n.agent for n in self.nodes.values())
        node_count = len(self.nodes)
        edge_count = len(self.edges)

        # Calculate max depth (longest chain from any root node)
        max_depth = self._calculate_max_depth()

        # Calculate average branching factor
        avg_branching = edge_count / node_count if node_count > 0 else 0.0

        # Calculate complexity score (0-1, normalized)
        # Based on: nodes, edges, depth, and rebuttals (indicates back-and-forth)
        rounds = len(set(n.round_num for n in self.nodes.values()))
        rebuttal_count = node_types.get("rebuttal", 0) + node_types.get("critique", 0)
        claim_count = node_types.get("proposal", 0) + node_types.get("evidence", 0)

        # Complexity formula: weighted combination of factors
        depth_factor = min(max_depth / 10.0, 1.0)  # Cap at depth 10
        branch_factor = min(avg_branching / 3.0, 1.0)  # Cap at 3 branches avg
        exchange_factor = min(rebuttal_count / (node_count + 1), 1.0)  # Ratio of rebuttals
        size_factor = min(node_count / 50.0, 1.0)  # Cap at 50 nodes

        complexity_score = (depth_factor * 0.25 + branch_factor * 0.25 +
                           exchange_factor * 0.3 + size_factor * 0.2)

        return {
            # Frontend GraphStats fields
            "node_count": node_count,
            "edge_count": edge_count,
            "max_depth": max_depth,
            "avg_branching": round(avg_branching, 2),
            "complexity_score": round(complexity_score, 3),
            "claim_count": claim_count,
            "rebuttal_count": rebuttal_count,
            # Additional detail fields
            "node_types": node_types,
            "edge_types": edge_types,
            "agents": list(agents),
            "rounds": rounds,
        }

    def _calculate_max_depth(self) -> int:
        """Calculate the maximum depth of the argument graph."""
        if not self.nodes:
            return 0

        # Build adjacency list for traversal
        children: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        has_parent = set()
        for edge in self.edges:
            if edge.source_id in children:
                children[edge.source_id].append(edge.target_id)
                has_parent.add(edge.target_id)

        # Find root nodes (nodes with no incoming edges)
        roots = [node_id for node_id in self.nodes if node_id not in has_parent]
        if not roots:
            # Cycle or no clear roots - use first node if available
            if not self.nodes:
                logger.warning("mapper_empty_nodes")
                return 0
            roots = [next(iter(self.nodes))]

        # BFS to find max depth (using deque for O(1) popleft)
        max_depth = 0
        visited = set()
        queue = deque((root, 1) for root in roots)

        while queue:
            node_id, depth = queue.popleft()  # O(1) vs O(n) for list.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            max_depth = max(max_depth, depth)

            for child_id in children.get(node_id, []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        return max_depth

    # --- Private helper methods ---

    def _make_id(self, agent: str, round_num: int, content: str) -> str:
        """Generate a unique, Mermaid-safe node ID."""
        h = hashlib.sha256(f"{agent}{round_num}{content[:50]}".encode()).hexdigest()[:8]
        safe_agent = agent[:3].lower().replace("-", "").replace("_", "")
        return f"{safe_agent}_{round_num}_{h}"

    def _infer_type(self, role: str, content: str) -> NodeType:
        """Infer node type from role and content heuristics."""
        content_lower = content.lower()
        
        # Role-based inference
        if role == "proposer":
            return NodeType.PROPOSAL
        if role == "critic":
            return NodeType.CRITIQUE
        
        # Content-based inference
        proposal_signals = ["i propose", "my proposal", "we should", "let's implement"]
        critique_signals = ["i disagree", "however", "issue with", "problem with", "concern"]
        concession_signals = ["i agree", "good point", "you're right", "valid point", "i concede"]
        rebuttal_signals = ["but", "on the contrary", "actually", "in response"]
        evidence_signals = ["evidence", "data shows", "according to", "research indicates"]
        
        if any(s in content_lower for s in proposal_signals):
            return NodeType.PROPOSAL
        if any(s in content_lower for s in concession_signals):
            return NodeType.CONCESSION
        if any(s in content_lower for s in critique_signals):
            return NodeType.CRITIQUE
        if any(s in content_lower for s in rebuttal_signals):
            return NodeType.REBUTTAL
        if any(s in content_lower for s in evidence_signals):
            return NodeType.EVIDENCE
        
        return NodeType.EVIDENCE  # Default fallback

    def _link_node(self, node: ArgumentNode, agent: str, round_num: int) -> None:
        """Create appropriate edges for a newly added node."""
        node_id = node.id
        
        if node.node_type == NodeType.PROPOSAL:
            self._last_proposal_id = node_id
            self._round_proposals[round_num] = node_id
            
        elif node.node_type in (NodeType.CRITIQUE, NodeType.REBUTTAL):
            # Link critiques/rebuttals to the round's proposal
            if round_num in self._round_proposals:
                self.edges.append(ArgumentEdge(
                    source_id=node_id,
                    target_id=self._round_proposals[round_num],
                    relation=EdgeRelation.REFUTES,
                ))
                
        elif node.node_type == NodeType.CONCESSION:
            # Link concessions to the last proposal
            if self._last_proposal_id:
                self.edges.append(ArgumentEdge(
                    source_id=node_id,
                    target_id=self._last_proposal_id,
                    relation=EdgeRelation.CONCEDES_TO,
                ))
                
        elif node.node_type == NodeType.EVIDENCE:
            # Link evidence to the agent's previous node (supporting their argument)
            if agent in self._agent_last_node:
                prev_id = self._agent_last_node[agent]
                if prev_id != node_id:
                    self.edges.append(ArgumentEdge(
                        source_id=node_id,
                        target_id=prev_id,
                        relation=EdgeRelation.SUPPORTS,
                    ))
        
        # Update agent's last node
        self._agent_last_node[agent] = node_id

    def _sanitize_for_mermaid(self, text: str) -> str:
        """Sanitize text for safe inclusion in Mermaid diagrams."""
        # Remove characters that break Mermaid syntax
        return (
            text.replace('"', "'")
                .replace("\n", " ")
                .replace("[", "(")
                .replace("]", ")")
                .replace("{", "(")
                .replace("}", ")")
                .replace("<", "‹")
                .replace(">", "›")
                .strip()
        )

    def _get_mermaid_arrow(self, relation: EdgeRelation) -> str:
        """Get the appropriate Mermaid arrow style for a relation."""
        arrows = {
            EdgeRelation.SUPPORTS: "-->",
            EdgeRelation.REFUTES: "-.->",
            EdgeRelation.MODIFIES: "-.->",
            EdgeRelation.RESPONDS_TO: "-->",
            EdgeRelation.CONCEDES_TO: "==>",
        }
        base = arrows.get(relation, "-->")
        
        # Add label for non-support relationships
        if relation not in (EdgeRelation.SUPPORTS, EdgeRelation.RESPONDS_TO):
            return f"{base[:-1]}|{relation.value}|{base[-1]}"
        return base
"""
Argument Cartographer - builds directed graphs of debate logic in real-time.

This is a pure observer that reads debate events and constructs a graph
representation. It never modifies debate state or agent prompts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
    full_content: str | None = None  # Store full text for detailed views
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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

    nodes: dict[str, ArgumentNode] = field(default_factory=dict)
    edges: list[ArgumentEdge] = field(default_factory=list)
    debate_id: str | None = None
    topic: str | None = None

    # Internal tracking for graph construction
    _last_proposal_id: str | None = None
    _agent_last_node: dict[str, str] = field(default_factory=dict)
    _round_proposals: dict[int, str] = field(default_factory=dict)

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
        metadata: dict[str, Any] | None = None,
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

        self._emit_graph_update("message", node_id)

        return node_id

    def update_from_critique(
        self,
        critic_agent: str,
        target_agent: str,
        severity: float,
        round_num: int,
        critique_text: str | None = None,
    ) -> str | None:
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

    def update_from_vote(self, agent: str, vote_value: str, round_num: int) -> str:
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
            self.edges.append(
                ArgumentEdge(
                    source_id=node_id,
                    target_id=self._round_proposals[round_num],
                    relation=EdgeRelation.RESPONDS_TO,
                )
            )

        self._emit_graph_update("vote", node_id)

        return node_id

    def update_from_consensus(
        self, result: str, round_num: int, vote_counts: dict[str, int] | None = None
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
                self.edges.append(
                    ArgumentEdge(
                        source_id=nid,
                        target_id=node_id,
                        relation=EdgeRelation.SUPPORTS,
                    )
                )

        self._emit_graph_update("consensus", node_id)

        return node_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph state as a dictionary.

        Used for emitting graph_update events over WebSocket so the
        frontend can render the argument graph in real time.
        """
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "statistics": self.get_statistics(),
        }

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
        rounds: dict[int, list[str]] = {}
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

        return json.dumps(
            {
                "debate_id": self.debate_id,
                "topic": self.topic,
                "nodes": nodes_data,
                "edges": [e.to_dict() for e in self.edges],
                "metadata": {
                    "node_count": len(self.nodes),
                    "edge_count": len(self.edges),
                    "exported_at": time.time(),
                },
            },
            indent=2,
            default=str,
        )

    def get_statistics(self) -> dict[str, Any]:
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

        complexity_score = (
            depth_factor * 0.25 + branch_factor * 0.25 + exchange_factor * 0.3 + size_factor * 0.2
        )

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
        children: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
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

    def _emit_graph_update(self, action: str, node_id: str) -> None:
        """Emit a graph update event for real-time streaming."""
        try:
            from aragora.events.dispatcher import dispatch_event

            dispatch_event(
                "argument_map_updated",
                {
                    "debate_id": self.debate_id or "",
                    "action": action,
                    "node_id": node_id,
                    "total_nodes": len(self.nodes),
                    "total_edges": len(self.edges),
                },
            )
        except Exception as e:  # noqa: BLE001
            # Broad catch: event emission is optional and must never disrupt
            # visualization. asyncpg errors can surface during webhook store init.
            logger.debug("Graph update event emission unavailable: %s", e)

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
                self.edges.append(
                    ArgumentEdge(
                        source_id=node_id,
                        target_id=self._round_proposals[round_num],
                        relation=EdgeRelation.REFUTES,
                    )
                )

        elif node.node_type == NodeType.CONCESSION:
            # Link concessions to the last proposal
            if self._last_proposal_id:
                self.edges.append(
                    ArgumentEdge(
                        source_id=node_id,
                        target_id=self._last_proposal_id,
                        relation=EdgeRelation.CONCEDES_TO,
                    )
                )

        elif node.node_type == NodeType.EVIDENCE:
            # Link evidence to the agent's previous node (supporting their argument)
            if agent in self._agent_last_node:
                prev_id = self._agent_last_node[agent]
                if prev_id != node_id:
                    self.edges.append(
                        ArgumentEdge(
                            source_id=node_id,
                            target_id=prev_id,
                            relation=EdgeRelation.SUPPORTS,
                        )
                    )

        # Update agent's last node
        self._agent_last_node[agent] = node_id

    def add_structural_annotation(
        self,
        node_id: str,
        annotation: dict[str, Any],
    ) -> bool:
        """
        Add a structural annotation to a node's metadata.

        Structural annotations come from the StructuralAnalyzer and include
        fallacy detections, premise chain data, and claim relationships.
        These are stored in the node's metadata under the "structural" key
        and also propagated to edge metadata when relevant.

        Args:
            node_id: The ID of the node to annotate
            annotation: Dict with structural analysis data (e.g. fallacies,
                premise_chains, claim_relationships)

        Returns:
            True if the annotation was added, False if the node was not found
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # Initialize structural annotations list if needed
        if "structural" not in node.metadata:
            node.metadata["structural"] = []

        node.metadata["structural"].append(annotation)

        # If annotation contains claim relationships, add corresponding edges
        claim_relationships = annotation.get("claim_relationships", [])
        for rel in claim_relationships:
            relation = rel.get("relation", "supports")
            # Map structural relation types to edge relations
            edge_relation_map = {
                "supports": EdgeRelation.SUPPORTS,
                "contradicts": EdgeRelation.REFUTES,
                "refines": EdgeRelation.MODIFIES,
            }
            edge_relation = edge_relation_map.get(relation, EdgeRelation.RESPONDS_TO)

            # Find a matching target node by content overlap
            target_claim = rel.get("target_claim", "")
            target_node_id = self._find_node_by_content(target_claim)
            if target_node_id and target_node_id != node_id:
                self.edges.append(
                    ArgumentEdge(
                        source_id=node_id,
                        target_id=target_node_id,
                        relation=edge_relation,
                        weight=rel.get("confidence", 0.5),
                        metadata={"source": "structural_analysis", "annotation": rel},
                    )
                )

        return True

    def _find_node_by_content(self, content_fragment: str) -> str | None:
        """Find a node whose content matches a fragment (for edge linking)."""
        if not content_fragment or len(content_fragment) < 10:
            return None

        content_lower = content_fragment.lower()
        best_match: str | None = None
        best_overlap = 0.0

        for node_id, node in self.nodes.items():
            node_text = (node.full_content or node.summary).lower()
            # Simple word overlap
            frag_words = {w for w in content_lower.split() if len(w) >= 3}
            node_words = {w for w in node_text.split() if len(w) >= 3}
            if not frag_words or not node_words:
                continue
            overlap = len(frag_words & node_words) / min(len(frag_words), len(node_words))
            if overlap > best_overlap and overlap > 0.3:
                best_overlap = overlap
                best_match = node_id

        return best_match

    def export_html(self, title: str = "Debate Argument Map") -> str:
        """
        Generate a self-contained HTML file with an interactive force-directed
        graph rendered on a canvas element using inline vanilla JavaScript.

        No external CDN dependencies -- works fully offline.

        Args:
            title: Page title shown in the browser tab and header.

        Returns:
            A complete HTML string that can be written to a file and opened
            in any modern browser.
        """
        import html as html_mod

        # Serialise nodes and edges into JSON-safe lists for embedding
        nodes_js = []
        for node in self.nodes.values():
            nodes_js.append(
                {
                    "id": node.id,
                    "agent": node.agent,
                    "type": node.node_type.value,
                    "summary": node.summary,
                    "content": node.full_content or node.summary,
                    "round": node.round_num,
                }
            )

        edges_js = []
        for edge in self.edges:
            if edge.source_id in self.nodes and edge.target_id in self.nodes:
                edges_js.append(
                    {
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "relation": edge.relation.value,
                    }
                )

        safe_title = html_mod.escape(title)
        topic_display = html_mod.escape(self.topic or title)
        graph_json = json.dumps({"nodes": nodes_js, "edges": edges_js})

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#1a1a2e; color:#eee; }}
header {{ padding:16px 24px; background:#16213e; border-bottom:2px solid #0f3460; }}
header h1 {{ font-size:20px; }}
#container {{ position:relative; width:100%; height:calc(100vh - 120px); }}
canvas {{ display:block; width:100%; height:100%; cursor:grab; }}
canvas.dragging {{ cursor:grabbing; }}
#legend {{ display:flex; gap:14px; flex-wrap:wrap; padding:10px 24px; background:#16213e; border-top:1px solid #0f3460; position:fixed; bottom:0; width:100%; }}
.leg {{ display:flex; align-items:center; gap:6px; font-size:13px; }}
.leg span.dot {{ width:14px; height:14px; border-radius:50%; display:inline-block; }}
#tooltip {{ position:absolute; display:none; background:#16213e; border:1px solid #0f3460;
  border-radius:6px; padding:12px 16px; max-width:360px; font-size:13px;
  line-height:1.45; pointer-events:none; z-index:10; box-shadow:0 4px 12px rgba(0,0,0,.4); }}
#tooltip .tt-agent {{ color:#aaa; font-size:11px; text-transform:uppercase; margin-bottom:4px; }}
#tooltip .tt-type {{ display:inline-block; padding:2px 8px; border-radius:3px; font-size:11px; margin-bottom:6px; }}
#detail {{ position:absolute; top:60px; right:16px; width:340px; max-height:60vh; overflow-y:auto;
  background:#16213e; border:1px solid #0f3460; border-radius:8px; padding:16px; display:none;
  font-size:13px; line-height:1.5; z-index:20; box-shadow:0 4px 16px rgba(0,0,0,.5); }}
#detail .close {{ float:right; cursor:pointer; color:#888; font-size:18px; }}
#detail .close:hover {{ color:#fff; }}
#detail h3 {{ margin-bottom:8px; font-size:15px; }}
#detail pre {{ white-space:pre-wrap; word-break:break-word; background:#0f1a30; padding:10px; border-radius:4px; margin-top:8px; }}
</style>
</head>
<body>
<header><h1>{topic_display}</h1></header>
<div id="container"><canvas id="c"></canvas></div>
<div id="tooltip"><div class="tt-agent"></div><div class="tt-type"></div><div class="tt-body"></div></div>
<div id="detail"><span class="close" onclick="this.parentElement.style.display='none'">&times;</span><h3 id="d-title"></h3><div id="d-meta"></div><pre id="d-content"></pre></div>
<div id="legend">
  <div class="leg"><span class="dot" style="background:#4488ff"></span>Proposal</div>
  <div class="leg"><span class="dot" style="background:#ee4444"></span>Critique</div>
  <div class="leg"><span class="dot" style="background:#44bb66"></span>Evidence</div>
  <div class="leg"><span class="dot" style="background:#ee8833"></span>Concession</div>
  <div class="leg"><span class="dot" style="background:#aa44dd"></span>Rebuttal</div>
  <div class="leg"><span class="dot" style="background:#999999"></span>Vote</div>
  <div class="leg"><span class="dot" style="background:#ddaa00"></span>Consensus</div>
</div>
<script>
(function() {{
  var DATA = {graph_json};
  var NODE_COLORS = {{proposal:"#4488ff",critique:"#ee4444",evidence:"#44bb66",concession:"#ee8833",rebuttal:"#aa44dd",vote:"#999999",consensus:"#ddaa00"}};
  var EDGE_STYLES = {{supports:{{color:"#44bb66",dash:[]}},refutes:{{color:"#ee4444",dash:[6,4]}},modifies:{{color:"#4488ff",dash:[2,4]}},responds_to:{{color:"#888888",dash:[]}},concedes_to:{{color:"#ee8833",dash:[4,3]}}}};
  var canvas=document.getElementById("c"), ctx=canvas.getContext("2d");
  var W, H, dpr=window.devicePixelRatio||1;
  function resize() {{ W=canvas.parentElement.clientWidth; H=canvas.parentElement.clientHeight; canvas.width=W*dpr; canvas.height=H*dpr; canvas.style.width=W+"px"; canvas.style.height=H+"px"; ctx.setTransform(dpr,0,0,dpr,0,0); }}
  resize(); window.addEventListener("resize", resize);

  var nodes=DATA.nodes.map(function(n,i) {{
    var angle=2*Math.PI*i/Math.max(DATA.nodes.length,1);
    var r=Math.min(W,H)*0.3;
    return {{id:n.id,agent:n.agent,type:n.type,summary:n.summary,content:n.content,round:n.round,
      x:W/2+r*Math.cos(angle)+Math.random()*40-20, y:H/2+r*Math.sin(angle)+Math.random()*40-20,
      vx:0, vy:0, radius:Math.min(8+n.summary.length/12,22), pinned:false}};
  }});
  var nodeMap={{}}; nodes.forEach(function(n){{ nodeMap[n.id]=n; }});
  var edges=DATA.edges.filter(function(e){{ return nodeMap[e.source]&&nodeMap[e.target]; }}).map(function(e){{ return {{source:nodeMap[e.source],target:nodeMap[e.target],relation:e.relation}}; }});

  var dragged=null, hovered=null, mx=0, my=0, offsetX=0, offsetY=0;
  var DAMPING=0.85, REPULSION=3000, SPRING=0.005, REST_LEN=120, CENTER_PULL=0.0003, DT=0.8;

  function tick() {{
    for(var i=0;i<nodes.length;i++) {{
      var a=nodes[i]; if(a.pinned) continue;
      a.vx+=(W/2-a.x)*CENTER_PULL; a.vy+=(H/2-a.y)*CENTER_PULL;
      for(var j=i+1;j<nodes.length;j++) {{
        var b=nodes[j], dx=a.x-b.x, dy=a.y-b.y, d2=dx*dx+dy*dy+1;
        var f=REPULSION/d2, fx=dx/Math.sqrt(d2)*f, fy=dy/Math.sqrt(d2)*f;
        a.vx+=fx; a.vy+=fy; if(!b.pinned){{ b.vx-=fx; b.vy-=fy; }}
      }}
    }}
    for(var i=0;i<edges.length;i++) {{
      var e=edges[i], dx=e.target.x-e.source.x, dy=e.target.y-e.source.y;
      var dist=Math.sqrt(dx*dx+dy*dy)+0.1, f=SPRING*(dist-REST_LEN);
      var fx=dx/dist*f, fy=dy/dist*f;
      if(!e.source.pinned){{ e.source.vx+=fx; e.source.vy+=fy; }}
      if(!e.target.pinned){{ e.target.vx-=fx; e.target.vy-=fy; }}
    }}
    for(var i=0;i<nodes.length;i++) {{
      var n=nodes[i]; if(n.pinned) continue;
      n.vx*=DAMPING; n.vy*=DAMPING;
      n.x+=n.vx*DT; n.y+=n.vy*DT;
      n.x=Math.max(n.radius,Math.min(W-n.radius,n.x));
      n.y=Math.max(n.radius,Math.min(H-n.radius,n.y));
    }}
  }}

  function drawArrow(x1,y1,x2,y2,r2,style) {{
    var dx=x2-x1,dy=y2-y1,dist=Math.sqrt(dx*dx+dy*dy)+.1;
    var ux=dx/dist,uy=dy/dist;
    var ex=x2-ux*(r2+4),ey=y2-uy*(r2+4);
    ctx.beginPath(); ctx.setLineDash(style.dash); ctx.strokeStyle=style.color; ctx.lineWidth=1.5;
    ctx.moveTo(x1,y1); ctx.lineTo(ex,ey); ctx.stroke(); ctx.setLineDash([]);
    var alen=8,aang=0.45;
    ctx.beginPath(); ctx.fillStyle=style.color;
    ctx.moveTo(ex,ey);
    ctx.lineTo(ex-alen*Math.cos(Math.atan2(uy,ux)-aang),ey-alen*Math.sin(Math.atan2(uy,ux)-aang));
    ctx.lineTo(ex-alen*Math.cos(Math.atan2(uy,ux)+aang),ey-alen*Math.sin(Math.atan2(uy,ux)+aang));
    ctx.closePath(); ctx.fill();
  }}

  function draw() {{
    ctx.clearRect(0,0,W,H);
    for(var i=0;i<edges.length;i++) {{
      var e=edges[i], st=EDGE_STYLES[e.relation]||EDGE_STYLES.responds_to;
      drawArrow(e.source.x,e.source.y,e.target.x,e.target.y,e.target.radius,st);
    }}
    for(var i=0;i<nodes.length;i++) {{
      var n=nodes[i], col=NODE_COLORS[n.type]||"#888";
      ctx.beginPath(); ctx.arc(n.x,n.y,n.radius,0,Math.PI*2);
      ctx.fillStyle=n===hovered?"#fff":col; ctx.fill();
      ctx.strokeStyle=n===dragged?"#fff":"rgba(255,255,255,.3)"; ctx.lineWidth=n===dragged?2.5:1; ctx.stroke();
      ctx.fillStyle="#fff"; ctx.font="10px sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle";
      var label=n.agent.length>6?n.agent.substring(0,5)+"..":n.agent;
      ctx.fillText(label,n.x,n.y+n.radius+12);
    }}
  }}

  function loop() {{ tick(); draw(); requestAnimationFrame(loop); }} loop();

  function nodeAt(px,py) {{
    for(var i=nodes.length-1;i>=0;i--) {{
      var n=nodes[i],dx=px-n.x,dy=py-n.y;
      if(dx*dx+dy*dy<=n.radius*n.radius) return n;
    }} return null;
  }}
  function pos(e) {{ var r=canvas.getBoundingClientRect(); return [e.clientX-r.left, e.clientY-r.top]; }}

  var tooltip=document.getElementById("tooltip");
  canvas.addEventListener("mousemove",function(e){{
    var p=pos(e); mx=p[0]; my=p[1];
    if(dragged){{ dragged.x=mx+offsetX; dragged.y=my+offsetY; return; }}
    var n=nodeAt(mx,my); hovered=n;
    if(n){{
      tooltip.querySelector(".tt-agent").textContent=n.agent+" (round "+n.round+")";
      var tb=tooltip.querySelector(".tt-type"); tb.textContent=n.type; tb.style.background=NODE_COLORS[n.type]||"#888";
      tooltip.querySelector(".tt-body").textContent=n.summary;
      tooltip.style.display="block";
      var tx=e.clientX+14,ty=e.clientY+14;
      if(tx+370>window.innerWidth) tx=e.clientX-374;
      tooltip.style.left=tx+"px"; tooltip.style.top=ty+"px";
    }} else {{ tooltip.style.display="none"; }}
  }});

  canvas.addEventListener("mousedown",function(e){{
    var p=pos(e), n=nodeAt(p[0],p[1]);
    if(n){{ dragged=n; n.pinned=true; offsetX=n.x-p[0]; offsetY=n.y-p[1]; canvas.classList.add("dragging"); e.preventDefault(); }}
  }});
  window.addEventListener("mouseup",function(){{
    if(dragged){{ dragged.pinned=false; dragged=null; canvas.classList.remove("dragging"); }}
  }});

  canvas.addEventListener("click",function(e){{
    var p=pos(e), n=nodeAt(p[0],p[1]);
    if(n){{
      var d=document.getElementById("detail");
      document.getElementById("d-title").textContent=n.agent+" - "+n.type;
      document.getElementById("d-meta").textContent="Round "+n.round;
      document.getElementById("d-content").textContent=n.content;
      d.style.display="block";
    }}
  }});
}})();
</script>
</body>
</html>"""

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

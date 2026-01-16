"""
Belief Network and Reasoning endpoint handlers.

Endpoints:
- GET /api/belief-network/:debate_id/cruxes - Get key claims that impact debate outcome
- GET /api/belief-network/:debate_id/load-bearing-claims - Get high-centrality claims
- GET /api/provenance/:debate_id/claims/:claim_id/support - Get claim verification status
- GET /api/laboratory/emergent-traits - Get emergent traits from agent performance
- GET /api/debate/:debate_id/graph-stats - Get argument graph statistics
"""

from __future__ import annotations

__all__ = [
    "BeliefHandler",
]

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

from aragora.server.validation import validate_debate_id, validate_id
from aragora.utils.optional_imports import try_import

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_clamped_int_param,
    handle_errors,
    json_response,
)
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for belief network endpoints (60 requests per minute - read-heavy)
_belief_limiter = RateLimiter(requests_per_minute=60)

# Lazy imports for optional dependencies using centralized utility
_belief_imports, BELIEF_NETWORK_AVAILABLE = try_import(
    "aragora.reasoning.belief", "BeliefNetwork", "BeliefPropagationAnalyzer"
)
BeliefNetwork = _belief_imports["BeliefNetwork"]
BeliefPropagationAnalyzer = _belief_imports["BeliefPropagationAnalyzer"]

_lab_imports, LABORATORY_AVAILABLE = try_import("aragora.agents.laboratory", "PersonaLaboratory")
PersonaLaboratory = _lab_imports["PersonaLaboratory"]

_prov_imports, PROVENANCE_AVAILABLE = try_import(
    "aragora.reasoning.provenance", "ProvenanceTracker"
)
ProvenanceTracker = _prov_imports["ProvenanceTracker"]


class BeliefHandler(BaseHandler):
    """Handler for belief network and reasoning endpoints."""

    ROUTES: list[str] = [
        # Note: /api/laboratory/emergent-traits handled by LaboratoryHandler
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle dynamic routes
        if path.startswith("/api/belief-network/") and path.endswith("/cruxes"):
            return True
        if path.startswith("/api/belief-network/") and path.endswith("/load-bearing-claims"):
            return True
        if path.startswith("/api/belief-network/") and path.endswith("/graph"):
            return True
        if path.startswith("/api/belief-network/") and path.endswith("/export"):
            return True
        if "/claims/" in path and path.endswith("/support"):
            return True
        if path.startswith("/api/debate/") and path.endswith("/graph-stats"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route belief network requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _belief_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for belief endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Get nomic_dir from server context
        nomic_dir = self.ctx.get("nomic_dir")
        # Note: /api/laboratory/emergent-traits handled by LaboratoryHandler

        if path.startswith("/api/belief-network/") and path.endswith("/cruxes"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            top_k = get_clamped_int_param(query_params, "top_k", 3, min_val=1, max_val=10)
            return self._get_debate_cruxes(nomic_dir, debate_id, top_k)

        if path.startswith("/api/belief-network/") and path.endswith("/load-bearing-claims"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            limit = get_clamped_int_param(query_params, "limit", 5, min_val=1, max_val=20)
            return self._get_load_bearing_claims(nomic_dir, debate_id, limit)

        if path.startswith("/api/belief-network/") and path.endswith("/graph"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            include_cruxes = query_params.get("include_cruxes", ["true"])[0].lower() == "true"
            return self._get_belief_network_graph(nomic_dir, debate_id, include_cruxes)

        if path.startswith("/api/belief-network/") and path.endswith("/export"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            format_type = query_params.get("format", ["json"])[0].lower()
            return self._export_belief_network(nomic_dir, debate_id, format_type)

        if "/claims/" in path and path.endswith("/support"):
            # Pattern: /api/provenance/:debate_id/claims/:claim_id/support
            parts = path.split("/")
            if len(parts) >= 6:
                debate_id = parts[3]
                claim_id = parts[5]
                valid_debate, _ = validate_debate_id(debate_id)
                valid_claim, _ = validate_id(claim_id, "claim ID")
                if not valid_debate or not valid_claim:
                    return error_response("Invalid ID format", 400)
                return self._get_claim_support(nomic_dir, debate_id, claim_id)
            return error_response("Invalid path format", 400)

        if path.startswith("/api/debate/") and path.endswith("/graph-stats"):
            debate_id = self._extract_debate_id(path, 3)
            if debate_id is None:
                return error_response("Invalid debate_id", 400)
            return self._get_debate_graph_stats(nomic_dir, debate_id)

        return None

    def _extract_debate_id(self, path: str, segment_index: int) -> Optional[str]:
        """Extract and validate debate ID from path."""
        parts = path.split("/")
        if len(parts) > segment_index:
            debate_id = parts[segment_index]
            is_valid, _ = validate_debate_id(debate_id)
            if is_valid:
                return debate_id
        return None

    @handle_errors("emergent traits retrieval")
    def _get_emergent_traits(
        self, nomic_dir: Optional[Path], persona_manager, min_confidence: float, limit: int
    ) -> HandlerResult:
        """Get emergent traits detected from agent performance patterns."""
        if not LABORATORY_AVAILABLE:
            return error_response("Persona laboratory not available", 503)

        lab = PersonaLaboratory(
            db_path=str(nomic_dir / "laboratory.db") if nomic_dir else None,
            persona_manager=persona_manager,
        )
        traits = lab.detect_emergent_traits()
        filtered = [t for t in traits if t.confidence >= min_confidence][:limit]
        return json_response(
            {
                "emergent_traits": [
                    {
                        "agent": t.agent_name,
                        "trait": t.trait_name,
                        "domain": t.domain,
                        "confidence": t.confidence,
                        "evidence": t.evidence,
                        "detected_at": t.detected_at,
                    }
                    for t in filtered
                ],
                "count": len(filtered),
                "min_confidence": min_confidence,
            }
        )

    @handle_errors("debate cruxes retrieval")
    def _get_debate_cruxes(
        self, nomic_dir: Optional[Path], debate_id: str, top_k: int
    ) -> HandlerResult:
        """Get key claims that would most impact the debate outcome."""
        if not BELIEF_NETWORK_AVAILABLE:
            return error_response("Belief network not available", 503)

        from aragora.debate.traces import DebateTrace

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            return error_response("Debate trace not found", 404)

        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()  # type: ignore[attr-defined]

        # Build belief network from debate
        network = BeliefNetwork(debate_id=debate_id)
        for msg in result.messages:
            network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

        analyzer = BeliefPropagationAnalyzer(network)
        cruxes = analyzer.identify_debate_cruxes(top_k=top_k)

        return json_response(
            {
                "debate_id": debate_id,
                "cruxes": cruxes,
                "count": len(cruxes),
            }
        )

    @handle_errors("load bearing claims retrieval")
    def _get_load_bearing_claims(
        self, nomic_dir: Optional[Path], debate_id: str, limit: int
    ) -> HandlerResult:
        """Get claims with highest centrality (most load-bearing)."""
        if not BELIEF_NETWORK_AVAILABLE:
            return error_response("Belief network not available", 503)

        from aragora.debate.traces import DebateTrace

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            return error_response("Debate trace not found", 404)

        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()  # type: ignore[attr-defined]

        # Build belief network from debate
        network = BeliefNetwork(debate_id=debate_id)
        for msg in result.messages:
            network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

        load_bearing = network.get_load_bearing_claims(limit=limit)

        return json_response(
            {
                "debate_id": debate_id,
                "load_bearing_claims": [
                    {
                        "claim_id": node.claim_id,
                        "statement": node.claim_statement,
                        "author": node.author,
                        "centrality": centrality,
                    }
                    for node, centrality in load_bearing
                ],
                "count": len(load_bearing),
            }
        )

    @handle_errors("claim support retrieval")
    def _get_claim_support(
        self, nomic_dir: Optional[Path], debate_id: str, claim_id: str
    ) -> HandlerResult:
        """Get verification status of all evidence supporting a claim."""
        if not PROVENANCE_AVAILABLE:
            return error_response("Provenance tracker not available", 503)

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        provenance_path = nomic_dir / "provenance" / f"{debate_id}.json"
        if not provenance_path.exists():
            return json_response(
                {
                    "debate_id": debate_id,
                    "claim_id": claim_id,
                    "support": None,
                    "message": "No provenance data for this debate",
                }
            )

        tracker = ProvenanceTracker.load(provenance_path)
        support = tracker.get_claim_support(claim_id)

        return json_response(
            {
                "debate_id": debate_id,
                "claim_id": claim_id,
                "support": support,
            }
        )

    @handle_errors("debate graph stats retrieval")
    def _get_debate_graph_stats(self, nomic_dir: Optional[Path], debate_id: str) -> HandlerResult:
        """Get argument graph statistics for a debate."""
        from aragora.debate.traces import DebateTrace
        from aragora.visualization.mapper import ArgumentCartographer

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            # Try replays directory as fallback
            replay_path = nomic_dir / "replays" / debate_id / "events.jsonl"
            if replay_path.exists():
                cartographer = ArgumentCartographer()
                cartographer.set_debate_context(debate_id, "")
                with replay_path.open() as f:
                    for line in f:
                        if line.strip():
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue  # Skip malformed event lines
                            if event.get("type") == "agent_message":
                                cartographer.update_from_message(
                                    agent=event.get("agent", "unknown"),
                                    content=event.get("data", {}).get("content", ""),
                                    role=event.get("data", {}).get("role", "proposer"),
                                    round_num=event.get("round", 1),
                                )
                            elif event.get("type") == "critique":
                                cartographer.update_from_critique(
                                    critic_agent=event.get("agent", "unknown"),
                                    target_agent=event.get("data", {}).get("target", "unknown"),
                                    severity=event.get("data", {}).get("severity", 0.5),
                                    round_num=event.get("round", 1),
                                    critique_text=event.get("data", {}).get("content", ""),
                                )
                stats = cartographer.get_statistics()
                return json_response(stats)
            else:
                return error_response("Debate not found", 404)

        # Load from trace file
        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()  # type: ignore[attr-defined]

        # Build cartographer from debate result
        cartographer = ArgumentCartographer()
        cartographer.set_debate_context(debate_id, result.task or "")

        for msg in result.messages:
            cartographer.update_from_message(
                agent=msg.agent,
                content=msg.content,
                role=msg.role,
                round_num=msg.round,
            )

        for critique in result.critiques:
            cartographer.update_from_critique(
                critic_agent=critique.agent,
                target_agent=critique.target or "",
                severity=critique.severity,
                round_num=getattr(critique, "round", 1),
                critique_text=critique.reasoning,
            )

        stats = cartographer.get_statistics()
        return json_response(stats)

    @handle_errors("belief network graph retrieval")
    def _get_belief_network_graph(
        self, nomic_dir: Optional[Path], debate_id: str, include_cruxes: bool = True
    ) -> HandlerResult:
        """Get belief network as a graph structure for visualization.

        Returns nodes (claims) and links (influence relationships) suitable
        for force-directed graph rendering.

        Response:
        {
            "nodes": [
                {
                    "id": "claim_001",
                    "claim_id": "claim_001",
                    "statement": "...",
                    "author": "claude",
                    "centrality": 0.85,
                    "is_crux": true,
                    "crux_score": 0.92,
                    "entropy": 0.65,
                    "belief": {"true_prob": 0.6, "false_prob": 0.2, "uncertain_prob": 0.2}
                }
            ],
            "links": [
                {"source": "claim_001", "target": "claim_002", "weight": 0.7, "type": "supports"}
            ],
            "metadata": {
                "debate_id": "debate_abc",
                "total_claims": 15,
                "crux_count": 3
            }
        }
        """
        if not BELIEF_NETWORK_AVAILABLE:
            return error_response("Belief network not available", 503)

        from aragora.debate.traces import DebateTrace

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            return error_response("Debate trace not found", 404)

        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()  # type: ignore[attr-defined]

        # Build belief network
        network = BeliefNetwork(debate_id=debate_id)
        for msg in result.messages:
            network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

        # Get cruxes if requested
        crux_ids = set()
        crux_scores = {}
        if include_cruxes:
            analyzer = BeliefPropagationAnalyzer(network)
            cruxes = analyzer.identify_debate_cruxes(top_k=10)
            for crux in cruxes:
                crux_ids.add(crux.get("claim_id", ""))
                crux_scores[crux.get("claim_id", "")] = crux.get("crux_score", 0)

        # Build graph structure
        nodes = []
        node_ids = set()

        for node_data in network.get_all_claims():
            node = node_data.get("node")
            if not node:
                continue

            claim_id = node.claim_id
            node_ids.add(claim_id)

            belief = (
                node.get_belief_distribution() if hasattr(node, "get_belief_distribution") else None
            )

            nodes.append(
                {
                    "id": claim_id,
                    "claim_id": claim_id,
                    "statement": node.claim_statement,
                    "author": node.author,
                    "centrality": node_data.get("centrality", 0.5),
                    "is_crux": claim_id in crux_ids,
                    "crux_score": crux_scores.get(claim_id),
                    "entropy": node_data.get("entropy", 0.5),
                    "belief": belief,
                }
            )

        # Build links from influence relationships
        links = []
        for edge in network.get_all_edges():
            source = edge.get("source")
            target = edge.get("target")
            if source in node_ids and target in node_ids:
                links.append(
                    {
                        "source": source,
                        "target": target,
                        "weight": edge.get("weight", 0.5),
                        "type": edge.get("type", "influences"),
                    }
                )

        return json_response(
            {
                "nodes": nodes,
                "links": links,
                "metadata": {
                    "debate_id": debate_id,
                    "total_claims": len(nodes),
                    "crux_count": len(crux_ids),
                },
            }
        )

    @handle_errors("belief network export")
    def _export_belief_network(
        self, nomic_dir: Optional[Path], debate_id: str, format_type: str = "json"
    ) -> HandlerResult:
        """Export belief network in various formats.

        Supported formats:
        - json: Full JSON structure (default)
        - graphml: GraphML format for Gephi/yEd
        - csv: CSV format (nodes and edges as separate arrays)

        Response varies by format type.
        """
        if not BELIEF_NETWORK_AVAILABLE:
            return error_response("Belief network not available", 503)

        from aragora.debate.traces import DebateTrace

        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            return error_response("Debate trace not found", 404)

        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()  # type: ignore[attr-defined]

        # Build belief network
        network = BeliefNetwork(debate_id=debate_id)
        for msg in result.messages:
            network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

        # Get cruxes
        analyzer = BeliefPropagationAnalyzer(network)
        cruxes = analyzer.identify_debate_cruxes(top_k=10)
        crux_ids = {c.get("claim_id", "") for c in cruxes}

        # Build export data
        nodes_data = []
        edges_data = []
        node_ids = set()

        for node_data in network.get_all_claims():
            node = node_data.get("node")
            if not node:
                continue

            claim_id = node.claim_id
            node_ids.add(claim_id)
            nodes_data.append(
                {
                    "id": claim_id,
                    "statement": node.claim_statement,
                    "author": node.author,
                    "centrality": node_data.get("centrality", 0.5),
                    "is_crux": claim_id in crux_ids,
                }
            )

        for edge in network.get_all_edges():
            source = edge.get("source")
            target = edge.get("target")
            if source in node_ids and target in node_ids:
                edges_data.append(
                    {
                        "source": source,
                        "target": target,
                        "weight": edge.get("weight", 0.5),
                        "type": edge.get("type", "influences"),
                    }
                )

        if format_type == "csv":
            # Return CSV-friendly structure
            return json_response(
                {
                    "format": "csv",
                    "debate_id": debate_id,
                    "nodes_csv": nodes_data,
                    "edges_csv": edges_data,
                    "headers": {
                        "nodes": ["id", "statement", "author", "centrality", "is_crux"],
                        "edges": ["source", "target", "weight", "type"],
                    },
                }
            )

        elif format_type == "graphml":
            # Build GraphML XML
            graphml_lines = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
                '  <key id="statement" for="node" attr.name="statement" attr.type="string"/>',
                '  <key id="author" for="node" attr.name="author" attr.type="string"/>',
                '  <key id="centrality" for="node" attr.name="centrality" attr.type="double"/>',
                '  <key id="is_crux" for="node" attr.name="is_crux" attr.type="boolean"/>',
                '  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>',
                '  <key id="type" for="edge" attr.name="type" attr.type="string"/>',
                f'  <graph id="{debate_id}" edgedefault="directed">',
            ]

            for node in nodes_data:
                statement_escaped = (
                    node["statement"]
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                graphml_lines.append(f'    <node id="{node["id"]}">')
                graphml_lines.append(f'      <data key="statement">{statement_escaped}</data>')
                graphml_lines.append(f'      <data key="author">{node["author"]}</data>')
                graphml_lines.append(f'      <data key="centrality">{node["centrality"]}</data>')
                graphml_lines.append(
                    f'      <data key="is_crux">{str(node["is_crux"]).lower()}</data>'
                )
                graphml_lines.append("    </node>")

            for i, edge in enumerate(edges_data):
                graphml_lines.append(
                    f'    <edge id="e{i}" source="{edge["source"]}" target="{edge["target"]}">'
                )
                graphml_lines.append(f'      <data key="weight">{edge["weight"]}</data>')
                graphml_lines.append(f'      <data key="type">{edge["type"]}</data>')
                graphml_lines.append("    </edge>")

            graphml_lines.extend(["  </graph>", "</graphml>"])

            return json_response(
                {
                    "format": "graphml",
                    "debate_id": debate_id,
                    "content": "\n".join(graphml_lines),
                    "content_type": "application/xml",
                }
            )

        else:
            # Default JSON format
            return json_response(
                {
                    "format": "json",
                    "debate_id": debate_id,
                    "nodes": nodes_data,
                    "edges": edges_data,
                    "summary": {
                        "total_nodes": len(nodes_data),
                        "total_edges": len(edges_data),
                        "crux_count": len(crux_ids),
                    },
                }
            )

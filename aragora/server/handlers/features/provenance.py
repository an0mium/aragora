"""
Provenance API handlers for decision tracing and audit trails.

Provides endpoints for:
- Retrieving debate provenance graphs
- Exporting provenance for compliance
- Verifying evidence chain integrity
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from aragora.reasoning.provenance import (
    MerkleTree,
    ProvenanceManager,
    SourceType,
)
from aragora.server.handlers.base import HandlerResult, json_response

logger = logging.getLogger(__name__)

# Global provenance store keyed by debate_id
# In production, this would be backed by persistent storage
_provenance_managers: Dict[str, ProvenanceManager] = {}


def get_provenance_manager(debate_id: str) -> ProvenanceManager:
    """Get or create a ProvenanceManager for a debate."""
    if debate_id not in _provenance_managers:
        _provenance_managers[debate_id] = ProvenanceManager(debate_id=debate_id)
    return _provenance_managers[debate_id]


def register_provenance_manager(debate_id: str, manager: ProvenanceManager) -> None:
    """Register an externally created ProvenanceManager."""
    _provenance_managers[debate_id] = manager


def _build_graph_nodes(manager: ProvenanceManager) -> List[Dict[str, Any]]:
    """Build visualization nodes from provenance records."""
    nodes = []

    for record in manager.chain.records:
        node = {
            "id": record.id,
            "type": _map_source_to_node_type(record.source_type),
            "label": _truncate(record.content, 100),
            "content": record.content,
            "source_type": record.source_type.value,
            "source_id": record.source_id,
            "timestamp": record.timestamp.isoformat(),
            "content_hash": record.content_hash[:16],
            "verified": record.verified,
            "confidence": record.confidence,
            "transformation": record.transformation.value,
        }
        nodes.append(node)

    return nodes


def _build_graph_edges(manager: ProvenanceManager) -> List[Dict[str, Any]]:
    """Build visualization edges from provenance relationships."""
    edges: List[Dict[str, Any]] = []

    # Chain links (previous_hash relationships)
    for i, record in enumerate(manager.chain.records):
        if record.previous_hash and i > 0:
            prev_record = manager.chain.records[i - 1]
            edges.append(
                {
                    "id": f"chain_{prev_record.id}_{record.id}",
                    "source": prev_record.id,
                    "target": record.id,
                    "type": "chain",
                    "label": "precedes",
                }
            )

        # Parent relationships (synthesis)
        for parent_id in record.parent_ids:
            edges.append(
                {
                    "id": f"parent_{parent_id}_{record.id}",
                    "source": parent_id,
                    "target": record.id,
                    "type": "synthesis",
                    "label": "synthesizes",
                }
            )

    # Citation relationships
    for citation in manager.graph.citations.values():
        edges.append(
            {
                "id": f"cite_{citation.claim_id}_{citation.evidence_id}",
                "source": citation.evidence_id,
                "target": citation.claim_id,
                "type": citation.support_type,
                "label": citation.support_type,
                "relevance": citation.relevance,
            }
        )

    return edges


def _map_source_to_node_type(source_type: SourceType) -> str:
    """Map source type to visualization node type."""
    mapping = {
        SourceType.AGENT_GENERATED: "argument",
        SourceType.USER_PROVIDED: "user_input",
        SourceType.EXTERNAL_API: "evidence",
        SourceType.WEB_SEARCH: "evidence",
        SourceType.DOCUMENT: "evidence",
        SourceType.CODE_ANALYSIS: "evidence",
        SourceType.DATABASE: "evidence",
        SourceType.COMPUTATION: "evidence",
        SourceType.SYNTHESIS: "synthesis",
        SourceType.AUDIO_TRANSCRIPT: "evidence",
        SourceType.UNKNOWN: "unknown",
    }
    return mapping.get(source_type, "unknown")


def _truncate(text: str, max_length: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _compute_max_depth(manager: ProvenanceManager) -> int:
    """Compute the maximum depth of the provenance graph."""
    if not manager.chain.records:
        return 0

    # Simple approach: count chain length
    return len(manager.chain.records)


async def handle_get_debate_provenance(debate_id: str) -> HandlerResult:
    """Get full provenance graph for a debate.

    Returns a visualization-ready graph with:
    - Nodes: question, agents, arguments, evidence, votes, consensus
    - Edges: supports, contradicts, synthesizes relationships
    """
    manager = get_provenance_manager(debate_id)

    nodes = _build_graph_nodes(manager)
    edges = _build_graph_edges(manager)

    # Verify chain integrity
    chain_valid, errors = manager.verify_chain_integrity()

    graph_data: Dict[str, Any] = {
        "debate_id": debate_id,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "max_depth": _compute_max_depth(manager),
            "verified": chain_valid,
            "verification_errors": errors,
            "status": "ready",
            "genesis_hash": manager.chain.genesis_hash,
        },
    }

    return json_response(graph_data)


async def handle_get_provenance_timeline(
    debate_id: str,
    round_number: Optional[int] = None,
) -> HandlerResult:
    """Get provenance as a timeline view.

    Shows how consensus evolved round-by-round.
    """
    manager = get_provenance_manager(debate_id)

    # Group records by round (using metadata if available)
    rounds_data: Dict[int, List[Dict[str, Any]]] = {}

    for record in manager.chain.records:
        round_num = record.metadata.get("round_number", 0)
        if round_number is not None and round_num != round_number:
            continue

        if round_num not in rounds_data:
            rounds_data[round_num] = []

        rounds_data[round_num].append(
            {
                "id": record.id,
                "content": _truncate(record.content, 200),
                "source_type": record.source_type.value,
                "source_id": record.source_id,
                "timestamp": record.timestamp.isoformat(),
                "transformation": record.transformation.value,
            }
        )

    # Build timeline response
    rounds = [
        {"round": round_num, "records": records}
        for round_num, records in sorted(rounds_data.items())
    ]

    # Track agent positions over time
    agent_positions: Dict[str, List[Dict[str, Any]]] = {}
    for record in manager.chain.records:
        if record.source_type == SourceType.AGENT_GENERATED:
            agent_id = record.source_id
            if agent_id not in agent_positions:
                agent_positions[agent_id] = []
            agent_positions[agent_id].append(
                {
                    "round": record.metadata.get("round_number", 0),
                    "position": record.metadata.get("position", "neutral"),
                    "content_preview": _truncate(record.content, 50),
                }
            )

    timeline_data = {
        "debate_id": debate_id,
        "rounds": rounds,
        "consensus_evolution": [],  # Would need consensus tracking data
        "agent_positions": agent_positions,
        "total_records": len(manager.chain.records),
    }

    return json_response(timeline_data)


async def handle_verify_provenance_chain(
    debate_id: str,
) -> HandlerResult:
    """Verify the integrity of a debate's provenance chain.

    Checks:
    - Hash chain integrity
    - Content hash validity
    - Citation completeness
    """
    manager = get_provenance_manager(debate_id)

    # Verify chain integrity
    chain_valid, chain_errors = manager.verify_chain_integrity()

    # Check content hash validity for each record
    content_errors = []
    for record in manager.chain.records:
        computed = record._compute_hash()
        if computed != record.content_hash:
            content_errors.append(f"Content hash mismatch for record {record.id}")

    # Check citation completeness
    citation_errors = []
    for citation in manager.graph.citations.values():
        if not manager.chain.get_record(citation.evidence_id):
            citation_errors.append(f"Citation references missing evidence: {citation.evidence_id}")

    all_errors = chain_errors + content_errors + citation_errors

    verification_result = {
        "debate_id": debate_id,
        "chain_valid": chain_valid and len(content_errors) == 0,
        "content_valid": len(content_errors) == 0,
        "citations_complete": len(citation_errors) == 0,
        "errors": all_errors,
        "verified_at": datetime.now().isoformat(),
        "record_count": len(manager.chain.records),
        "citation_count": len(manager.graph.citations),
    }

    return json_response(verification_result)


async def handle_export_provenance_report(
    debate_id: str,
    format: str = "json",
    include_evidence: bool = True,
    include_chain: bool = True,
) -> HandlerResult:
    """Export provenance data for compliance reporting.

    Formats:
    - json: Full structured data
    - summary: Human-readable summary
    - audit: Compliance-focused format with hash chains
    """
    manager = get_provenance_manager(debate_id)

    # Build base report
    chain_valid, _ = manager.verify_chain_integrity()

    report: Dict[str, Any] = {
        "debate_id": debate_id,
        "format": format,
        "generated_at": datetime.now().isoformat(),
        "chain_genesis": manager.chain.genesis_hash,
        "chain_tip": (manager.chain.records[-1].chain_hash() if manager.chain.records else None),
        "record_count": len(manager.chain.records),
        "citation_count": len(manager.graph.citations),
        "verification_status": "verified" if chain_valid else "failed",
    }

    if format == "json" and include_evidence:
        report["records"] = [r.to_dict() for r in manager.chain.records]
        report["citations"] = [
            {
                "claim_id": c.claim_id,
                "evidence_id": c.evidence_id,
                "relevance": c.relevance,
                "support_type": c.support_type,
            }
            for c in manager.graph.citations.values()
        ]

    if format == "audit" or (format == "json" and include_chain):
        # Build hash chain for audit trail
        hash_chain = []
        for record in manager.chain.records:
            hash_chain.append(
                {
                    "id": record.id,
                    "content_hash": record.content_hash,
                    "chain_hash": record.chain_hash(),
                    "previous_hash": record.previous_hash,
                    "timestamp": record.timestamp.isoformat(),
                }
            )
        report["hash_chain"] = hash_chain

        # Compute Merkle root for batch verification
        if manager.chain.records:
            merkle = MerkleTree(manager.chain.records)
            report["merkle_root"] = merkle.root

    if format == "summary":
        # Human-readable summary
        report["summary"] = {
            "total_evidence_pieces": len(manager.chain.records),
            "unique_sources": len(set(r.source_id for r in manager.chain.records)),
            "source_types": list(set(r.source_type.value for r in manager.chain.records)),
            "time_span": (
                {
                    "start": manager.chain.records[0].timestamp.isoformat(),
                    "end": manager.chain.records[-1].timestamp.isoformat(),
                }
                if manager.chain.records
                else None
            ),
            "chain_integrity": "intact" if chain_valid else "compromised",
        }

    return json_response(report)


async def handle_get_claim_provenance(
    debate_id: str,
    claim_id: str,
) -> HandlerResult:
    """Get provenance for a specific claim in a debate.

    Returns:
    - The claim details
    - Supporting evidence with provenance
    - Contradicting evidence with provenance
    - Agent attributions
    """
    manager = get_provenance_manager(debate_id)

    # Get claim support analysis
    support_analysis = manager.get_claim_support(claim_id)

    # Get supporting and contradicting evidence
    supporting = manager.graph.get_supporting_evidence(claim_id)
    contradicting = manager.graph.get_contradicting_evidence(claim_id)

    # Build supporting evidence list with provenance
    supporting_evidence = []
    for citation in supporting:
        record = manager.chain.get_record(citation.evidence_id)
        if record:
            supporting_evidence.append(
                {
                    "evidence_id": citation.evidence_id,
                    "content": record.content,
                    "source_type": record.source_type.value,
                    "source_id": record.source_id,
                    "relevance": citation.relevance,
                    "citation_text": citation.citation_text,
                    "verified": record.verified,
                    "content_hash": record.content_hash[:16],
                }
            )

    # Build contradicting evidence list
    contradicting_evidence = []
    for citation in contradicting:
        record = manager.chain.get_record(citation.evidence_id)
        if record:
            contradicting_evidence.append(
                {
                    "evidence_id": citation.evidence_id,
                    "content": record.content,
                    "source_type": record.source_type.value,
                    "source_id": record.source_id,
                    "relevance": citation.relevance,
                    "citation_text": citation.citation_text,
                    "verified": record.verified,
                    "content_hash": record.content_hash[:16],
                }
            )

    # Get claim record if it exists
    claim_record = manager.chain.get_record(claim_id)

    claim_provenance = {
        "debate_id": debate_id,
        "claim_id": claim_id,
        "claim_text": claim_record.content if claim_record else "",
        "agent_id": claim_record.source_id if claim_record else "",
        "round_number": (claim_record.metadata.get("round_number", 0) if claim_record else 0),
        "supporting_evidence": supporting_evidence,
        "contradicting_evidence": contradicting_evidence,
        "synthesis_parents": (claim_record.parent_ids if claim_record else []),
        "provenance_chain": (
            [r.id for r in manager.chain.get_ancestry(claim_id)] if claim_record else []
        ),
        "verification_status": {
            "evidence_verified": support_analysis.get("verified_count", 0) > 0,
            "chain_valid": support_analysis.get("failed_count", 0) == 0,
            "support_score": support_analysis.get("support_score", 0.0),
        },
    }

    return json_response(claim_provenance)


async def handle_get_agent_contributions(
    debate_id: str,
    agent_id: Optional[str] = None,
) -> HandlerResult:
    """Get provenance-tracked contributions by agent(s).

    Shows which evidence and arguments each agent contributed,
    with full provenance chains.
    """
    manager = get_provenance_manager(debate_id)

    # Group contributions by agent
    agent_contributions: Dict[str, List[Dict[str, Any]]] = {}

    for record in manager.chain.records:
        if record.source_type != SourceType.AGENT_GENERATED:
            continue

        source = record.source_id
        if agent_id is not None and source != agent_id:
            continue

        if source not in agent_contributions:
            agent_contributions[source] = []

        contribution = {
            "id": record.id,
            "type": record.transformation.value,
            "content": _truncate(record.content, 200),
            "timestamp": record.timestamp.isoformat(),
            "round_number": record.metadata.get("round_number", 0),
            "parent_ids": record.parent_ids,
            "verified": record.verified,
        }
        agent_contributions[source].append(contribution)

    # Build contributions list
    contributions = [
        {"agent_id": aid, "contributions": contribs}
        for aid, contribs in agent_contributions.items()
    ]

    # Compute summary statistics
    total_arguments = sum(len(c["contributions"]) for c in contributions)
    total_syntheses = sum(
        1
        for c in contributions
        for contrib in c["contributions"]
        if contrib["type"] == "aggregated"
    )

    response = {
        "debate_id": debate_id,
        "agent_id": agent_id,
        "contributions": contributions,
        "summary": {
            "total_arguments": total_arguments,
            "total_evidence": len(manager.chain.records) - total_arguments,
            "total_syntheses": total_syntheses,
            "consensus_contributions": 0,  # Would need consensus tracking
            "unique_agents": len(agent_contributions),
        },
    }

    return json_response(response)


def register_provenance_routes(router: Any) -> None:
    """Register provenance routes with the server router."""

    async def get_debate_provenance(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        return await handle_get_debate_provenance(debate_id)

    async def get_provenance_timeline(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        round_number = request.query_params.get("round")
        if round_number:
            round_number = int(round_number)
        return await handle_get_provenance_timeline(debate_id, round_number)

    async def verify_provenance(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        return await handle_verify_provenance_chain(debate_id)

    async def export_provenance(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        format = request.query_params.get("format", "json")
        include_evidence = request.query_params.get("include_evidence", "true") == "true"
        include_chain = request.query_params.get("include_chain", "true") == "true"
        return await handle_export_provenance_report(
            debate_id, format, include_evidence, include_chain
        )

    async def get_claim_provenance(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        claim_id = request.path_params.get("claim_id", "")
        return await handle_get_claim_provenance(debate_id, claim_id)

    async def get_agent_contributions(request: Any) -> HandlerResult:
        debate_id = request.path_params.get("debate_id", "")
        agent_id = request.query_params.get("agent_id")
        return await handle_get_agent_contributions(debate_id, agent_id)

    # Register routes
    router.add_route("GET", "/api/debates/{debate_id}/provenance", get_debate_provenance)
    router.add_route(
        "GET",
        "/api/debates/{debate_id}/provenance/timeline",
        get_provenance_timeline,
    )
    router.add_route("GET", "/api/debates/{debate_id}/provenance/verify", verify_provenance)
    router.add_route("GET", "/api/debates/{debate_id}/provenance/export", export_provenance)
    router.add_route(
        "GET",
        "/api/debates/{debate_id}/claims/{claim_id}/provenance",
        get_claim_provenance,
    )
    router.add_route(
        "GET",
        "/api/debates/{debate_id}/provenance/contributions",
        get_agent_contributions,
    )

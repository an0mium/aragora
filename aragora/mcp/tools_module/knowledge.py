"""
MCP Tools for Knowledge Mound operations.

Provides tools for querying and managing the Knowledge Mound:
- query_knowledge: Search the knowledge graph
- store_knowledge: Add new knowledge nodes
- get_knowledge_stats: Get knowledge base statistics
- get_decision_receipt: Get a formal decision receipt
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def query_knowledge_tool(
    query: str,
    node_types: str = "all",
    min_confidence: float = 0.0,
    limit: int = 10,
    include_relationships: bool = False,
) -> Dict[str, Any]:
    """
    Query the Knowledge Mound for relevant information.

    Args:
        query: Search query text
        node_types: Comma-separated node types (fact, insight, claim, evidence, decision)
        min_confidence: Minimum confidence threshold (0-1)
        limit: Maximum results to return
        include_relationships: Whether to include related nodes

    Returns:
        Dict with nodes, count, and query metadata
    """
    results: List[Dict[str, Any]] = []

    try:
        from aragora.knowledge.mound.core import get_knowledge_mound  # type: ignore[attr-defined]

        mound = get_knowledge_mound()

        # Parse node types
        types_filter = None if node_types == "all" else node_types.split(",")

        # Query the mound
        nodes = await mound.query(
            query=query,
            node_types=types_filter,
            min_confidence=min_confidence,
            limit=limit,
        )

        for node in nodes:
            result = {
                "id": node.id,
                "content": node.content[:500] if len(node.content) > 500 else node.content,
                "node_type": node.node_type,
                "confidence": node.confidence,
                "tier": node.tier,
                "created_at": (
                    node.created_at.isoformat()
                    if hasattr(node.created_at, "isoformat")
                    else str(node.created_at)
                ),
                "topics": node.topics[:5] if node.topics else [],
            }

            if include_relationships and hasattr(node, "relationships"):
                result["relationships"] = [
                    {
                        "type": rel.type,
                        "target_id": rel.target_id,
                        "weight": rel.weight,
                    }
                    for rel in node.relationships[:5]
                ]

            results.append(result)

    except ImportError:
        logger.warning("Knowledge Mound not available")
    except Exception as e:
        logger.error(f"Knowledge query failed: {e}")
        return {
            "error": f"Query failed: {str(e)}",
            "query": query,
        }

    return {
        "nodes": results,
        "count": len(results),
        "query": query,
        "filters": {
            "node_types": node_types,
            "min_confidence": min_confidence,
        },
    }


async def store_knowledge_tool(
    content: str,
    node_type: str = "fact",
    confidence: float = 0.8,
    tier: str = "medium",
    topics: str = "",
    source_debate_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Store a new knowledge node in the Knowledge Mound.

    Args:
        content: The knowledge content to store
        node_type: Type of node (fact, insight, claim, evidence, decision)
        confidence: Confidence level (0-1)
        tier: Storage tier (fast, medium, slow, glacial)
        topics: Comma-separated topics
        source_debate_id: Optional source debate ID

    Returns:
        Dict with stored node ID and metadata
    """
    valid_types = {"fact", "insight", "claim", "evidence", "decision", "opinion"}
    valid_tiers = {"fast", "medium", "slow", "glacial"}

    if node_type not in valid_types:
        return {"error": f"Invalid node_type. Must be one of: {valid_types}"}

    if tier not in valid_tiers:
        return {"error": f"Invalid tier. Must be one of: {valid_tiers}"}

    if not 0 <= confidence <= 1:
        return {"error": "Confidence must be between 0 and 1"}

    try:
        from aragora.knowledge.mound.core import get_knowledge_mound  # type: ignore[attr-defined]

        mound = get_knowledge_mound()

        # Parse topics
        topics_list = [t.strip() for t in topics.split(",") if t.strip()] if topics else []

        # Store the node
        node_id = await mound.store(
            content=content,
            node_type=node_type,
            confidence=confidence,
            tier=tier,
            topics=topics_list,
            metadata=(
                {
                    "source_debate_id": source_debate_id,
                    "stored_via": "mcp_tool",
                }
                if source_debate_id
                else {"stored_via": "mcp_tool"}
            ),
        )

        return {
            "node_id": node_id,
            "stored": True,
            "node_type": node_type,
            "confidence": confidence,
            "tier": tier,
            "topics": topics_list,
        }

    except ImportError:
        logger.warning("Knowledge Mound not available")
        return {"error": "Knowledge Mound module not available"}
    except Exception as e:
        logger.error(f"Failed to store knowledge: {e}")
        return {"error": f"Store failed: {str(e)}"}


async def get_knowledge_stats_tool() -> Dict[str, Any]:
    """
    Get statistics about the Knowledge Mound.

    Returns:
        Dict with node counts, tier utilization, and health metrics
    """
    try:
        from aragora.knowledge.mound.core import get_knowledge_mound  # type: ignore[attr-defined]

        mound = get_knowledge_mound()
        stats = await mound.get_stats()

        return {
            "total_nodes": stats.get("total_nodes", 0),
            "total_relationships": stats.get("total_relationships", 0),
            "nodes_by_type": stats.get("nodes_by_type", {}),
            "nodes_by_tier": stats.get("nodes_by_tier", {}),
            "avg_confidence": stats.get("avg_confidence", 0),
            "stale_nodes_count": stats.get("stale_nodes_count", 0),
            "last_updated": stats.get("last_updated", "unknown"),
        }

    except ImportError:
        logger.warning("Knowledge Mound not available")
        return {
            "error": "Knowledge Mound module not available",
            "total_nodes": 0,
        }
    except Exception as e:
        logger.error(f"Failed to get knowledge stats: {e}")
        return {"error": f"Stats failed: {str(e)}"}


async def get_decision_receipt_tool(
    debate_id: str,
    format: str = "json",
    include_proofs: bool = True,
    include_evidence: bool = True,
) -> Dict[str, Any]:
    """
    Get a formal decision receipt for a completed debate.

    A decision receipt provides an auditable record of:
    - The question/decision made
    - Participating agents
    - Final consensus
    - Confidence level
    - Supporting evidence
    - Formal proofs (if available)

    Args:
        debate_id: ID of the debate
        format: Output format (json, markdown, pdf)
        include_proofs: Include formal verification proofs
        include_evidence: Include cited evidence

    Returns:
        Dict with the decision receipt
    """
    try:
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if not db:
            return {"error": "Debates database not available"}

        debate = db.get(debate_id)
        if not debate:
            return {"error": f"Debate {debate_id} not found"}

        # Build the receipt
        receipt: Dict[str, Any] = {
            "receipt_id": f"receipt_{debate_id}_{int(time.time())}",
            "debate_id": debate_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "question": debate.get("task", "Unknown"),
            "decision": {
                "answer": debate.get("final_answer", "No answer"),
                "consensus_reached": debate.get("consensus_reached", False),
                "confidence": debate.get("confidence", 0),
                "confidence_percent": f"{debate.get('confidence', 0) * 100:.1f}%",
            },
            "process": {
                "rounds_used": debate.get("rounds_used", 0),
                "agents": debate.get("participants", []),
                "protocol": debate.get("protocol", "standard"),
            },
        }

        # Add proofs if requested
        if include_proofs:
            receipt["proofs"] = debate.get("proofs", [])

        # Add evidence if requested
        if include_evidence:
            receipt["evidence"] = debate.get("evidence", [])

        # Add verification status
        if debate.get("verified"):
            receipt["verification"] = {
                "verified": True,
                "verification_method": debate.get("verification_method", "unknown"),
                "verified_at": debate.get("verified_at", "unknown"),
            }

        # Format conversion
        if format == "markdown":
            receipt["formatted"] = _format_receipt_markdown(receipt)
        elif format == "pdf":
            receipt["note"] = "PDF generation requires additional processing"

        return receipt

    except Exception as e:
        logger.error(f"Failed to generate decision receipt: {e}")
        return {"error": f"Receipt generation failed: {str(e)}"}


def _format_receipt_markdown(receipt: Dict[str, Any]) -> str:
    """Format a decision receipt as markdown."""
    md = f"""# Decision Receipt

**Receipt ID:** {receipt["receipt_id"]}
**Generated:** {receipt["generated_at"]}

## Decision

**Question:** {receipt["question"]}

**Answer:** {receipt["decision"]["answer"]}

- Consensus Reached: {"Yes" if receipt["decision"]["consensus_reached"] else "No"}
- Confidence: {receipt["decision"]["confidence_percent"]}

## Process

- Rounds Used: {receipt["process"]["rounds_used"]}
- Agents: {", ".join(receipt["process"]["agents"])}
- Protocol: {receipt["process"]["protocol"]}
"""

    if receipt.get("proofs"):
        md += f"\n## Proofs\n\n{len(receipt['proofs'])} formal proofs available\n"

    if receipt.get("evidence"):
        md += f"\n## Evidence\n\n{len(receipt['evidence'])} evidence items cited\n"

    if receipt.get("verification"):
        md += (
            f"\n## Verification\n\nVerified via {receipt['verification']['verification_method']}\n"
        )

    return md


# Export all tools
__all__ = [
    "query_knowledge_tool",
    "store_knowledge_tool",
    "get_knowledge_stats_tool",
    "get_decision_receipt_tool",
]

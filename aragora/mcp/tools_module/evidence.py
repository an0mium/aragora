"""
MCP Evidence Tools.

Evidence collection, citation, and verification.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


async def search_evidence_tool(
    query: str,
    sources: str = "all",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search for evidence across configured sources.

    Args:
        query: Search query
        sources: Comma-separated sources (arxiv, hackernews, reddit, all)
        limit: Max results per source

    Returns:
        Dict with evidence results
    """
    if not query:
        return {"error": "query is required"}

    limit = min(max(limit, 1), 50)
    results: List[Dict[str, Any]] = []

    try:
        from aragora.evidence.collector import EvidenceCollector

        collector = EvidenceCollector()
        source_list = None if sources == "all" else [s.strip() for s in sources.split(",")]

        # Use collect_evidence which is the actual API
        evidence_pack = await collector.collect_evidence(
            task=query,
            enabled_connectors=source_list,
        )

        for e in evidence_pack.snippets[:limit]:
            results.append(
                {
                    "id": e.id,
                    "title": e.title,
                    "source": e.source,
                    "url": e.url,
                    "snippet": e.snippet[:300] if e.snippet else "",
                    "score": e.reliability_score,
                    "published": str(e.fetched_at) if e.fetched_at else None,
                }
            )

    except ImportError:
        logger.warning("Evidence collector not available")
    except Exception as e:
        logger.warning(f"Evidence search failed: {e}")

    return {
        "query": query,
        "sources": sources,
        "results": results,
        "count": len(results),
    }


async def cite_evidence_tool(
    debate_id: str,
    evidence_id: str,
    message_index: int,
    citation_text: str = "",
) -> Dict[str, Any]:
    """
    Add a citation to evidence in a debate message.

    Args:
        debate_id: ID of the debate
        evidence_id: ID of the evidence to cite
        message_index: Index of the message to add citation to
        citation_text: Optional citation text

    Returns:
        Dict with citation status
    """
    if not debate_id or not evidence_id:
        return {"error": "debate_id and evidence_id are required"}

    try:
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if not db:
            return {"error": "Storage not available"}

        debate = db.get(debate_id)
        if not debate:
            return {"error": f"Debate {debate_id} not found"}

        # Add citation to debate metadata
        citations = debate.get("citations", [])
        citation = {
            "evidence_id": evidence_id,
            "message_index": message_index,
            "text": citation_text,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        citations.append(citation)

        # Update debate
        if hasattr(db, "update"):
            db.update(debate_id, {"citations": citations})

        return {
            "success": True,
            "debate_id": debate_id,
            "evidence_id": evidence_id,
            "message_index": message_index,
            "citation_count": len(citations),
        }

    except Exception as e:
        return {"error": f"Failed to add citation: {e}"}


async def verify_citation_tool(
    url: str,
) -> Dict[str, Any]:
    """
    Verify that a citation URL is valid and accessible.

    Args:
        url: URL to verify

    Returns:
        Dict with verification status and metadata
    """
    if not url:
        return {"error": "url is required"}

    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                return {
                    "url": url,
                    "valid": response.status == 200,
                    "status_code": response.status,
                    "content_type": response.headers.get("Content-Type", "unknown"),
                    "accessible": response.status < 400,
                }
    except asyncio.TimeoutError:
        return {"url": url, "valid": False, "error": "Timeout"}
    except Exception as e:
        return {"url": url, "valid": False, "error": str(e)}


__all__ = [
    "search_evidence_tool",
    "cite_evidence_tool",
    "verify_citation_tool",
]

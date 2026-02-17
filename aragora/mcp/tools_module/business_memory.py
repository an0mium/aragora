"""
MCP tools for business knowledge ingestion and retrieval.

Exposes the MemoryFabric's business knowledge capabilities as MCP tools
for use by external AI agents.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def store_business_knowledge_tool(
    content: str,
    doc_type: str = "general",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store business knowledge via the unified MemoryFabric.

    Args:
        content: The document or interaction text to store.
        doc_type: Type of document (invoice, policy, contract, meeting_notes, general).
        metadata: Additional metadata to associate with the item.

    Returns:
        Dict with stored status, surprise score, and systems written to.
    """
    try:
        from aragora.memory.fabric import MemoryFabric
        from aragora.memory.business import BusinessKnowledgeIngester

        fabric = MemoryFabric()
        ingester = BusinessKnowledgeIngester(fabric)
        result = await ingester.ingest_document(content, doc_type, metadata)
        return {
            "stored": result.stored,
            "systems_written": result.systems_written,
            "surprise_score": result.surprise_combined,
            "reason": result.reason,
        }
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Business knowledge store failed: %s", exc)
        return {"stored": False, "error": "Business memory system unavailable"}


async def query_business_knowledge_tool(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Query business knowledge from all memory systems.

    Args:
        query: The topic or question to search for.
        limit: Maximum number of results to return.

    Returns:
        Dict with results list, each containing content, source, and relevance.
    """
    try:
        from aragora.memory.fabric import MemoryFabric
        from aragora.memory.business import BusinessKnowledgeIngester

        fabric = MemoryFabric()
        ingester = BusinessKnowledgeIngester(fabric)
        results = await ingester.query_business_context(query, limit)
        return {
            "results": [
                {
                    "content": r.content,
                    "source": r.source_system,
                    "relevance": r.relevance,
                    "recency": r.recency,
                }
                for r in results
            ],
            "count": len(results),
        }
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Business knowledge query failed: %s", exc)
        return {"results": [], "count": 0, "error": "Business memory system unavailable"}


__all__ = ["store_business_knowledge_tool", "query_business_knowledge_tool"]

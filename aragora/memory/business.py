"""
Business Knowledge Ingestion for SME use cases.

Provides helpers to ingest business-specific knowledge (invoices, interactions,
process documents, compliance requirements) through the MemoryFabric with
surprise scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from aragora.memory.fabric import FabricResult, MemoryFabric, RememberResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BusinessDocument:
    """Parsed business document ready for ingestion."""

    content: str
    doc_type: str  # invoice, policy, contract, process, meeting_notes
    metadata: dict[str, Any]


class BusinessKnowledgeIngester:
    """Ingest business-specific knowledge through the MemoryFabric.

    Supports: invoice patterns, customer interactions, process documentation,
    compliance requirements, team decisions.
    """

    def __init__(self, fabric: MemoryFabric):
        self._fabric = fabric

    async def ingest_document(
        self,
        content: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> RememberResult:
        """Parse and store business document with surprise scoring.

        High surprise: new customer terms, policy changes, exceptions.
        Low surprise: routine invoices matching prior patterns.
        """
        meta = metadata or {}
        meta["doc_type"] = doc_type

        # Fetch existing context for this doc type to calibrate surprise
        existing = await self._fabric.query(
            f"{doc_type} business document",
            limit=5,
        )
        existing_text = "\n".join(r.content for r in existing)

        tagged = f"[{doc_type}] {content}"
        return await self._fabric.remember(
            content=tagged,
            source=f"business_{doc_type}",
            metadata=meta,
            existing_context=existing_text,
        )

    async def ingest_interaction(
        self,
        summary: str,
        participants: list[str],
        outcome: str,
    ) -> RememberResult:
        """Store business interaction (meeting, call, decision)."""
        content = (
            f"Interaction with {', '.join(participants)}: {summary}\n"
            f"Outcome: {outcome}"
        )
        return await self._fabric.remember(
            content=content,
            source="business_interaction",
            metadata={
                "doc_type": "interaction",
                "participants": participants,
            },
        )

    async def query_business_context(
        self,
        topic: str,
        limit: int = 10,
    ) -> list[FabricResult]:
        """Query business knowledge for debate context."""
        return await self._fabric.query(topic, limit=limit)


__all__ = ["BusinessKnowledgeIngester", "BusinessDocument"]

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
        content = f"Interaction with {', '.join(participants)}: {summary}\nOutcome: {outcome}"
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

    async def track_customer(
        self,
        customer_id: str,
        interaction: str,
        sentiment: float = 0.5,
    ) -> RememberResult:
        """Track a customer interaction with sentiment scoring.

        Unusual sentiment (very positive or very negative) gets higher
        surprise and is prioritized for retention.

        Args:
            customer_id: Unique customer identifier
            interaction: Description of the interaction
            sentiment: Sentiment score 0.0 (negative) to 1.0 (positive)

        Returns:
            RememberResult
        """
        content = f"[customer:{customer_id}] {interaction}\nSentiment: {sentiment:.2f}"

        # Fetch existing customer context for surprise calibration
        existing = await self._fabric.query(
            f"customer {customer_id}",
            limit=5,
        )
        existing_text = "\n".join(r.content for r in existing)

        return await self._fabric.remember(
            content=content,
            source="business_customer",
            metadata={
                "doc_type": "customer_interaction",
                "customer_id": customer_id,
                "sentiment": sentiment,
            },
            existing_context=existing_text,
        )

    async def record_decision(
        self,
        decision: str,
        context: str,
        outcome: str | None = None,
    ) -> RememberResult:
        """Record a business decision with its context and optional outcome.

        Args:
            decision: The decision that was made
            context: Context/reasoning behind the decision
            outcome: Optional outcome if known

        Returns:
            RememberResult
        """
        parts = [
            f"[decision] {decision}",
            f"Context: {context}",
        ]
        if outcome:
            parts.append(f"Outcome: {outcome}")
        content = "\n".join(parts)

        # Fetch existing decisions for surprise calibration
        existing = await self._fabric.query(
            f"decision {decision[:50]}",
            limit=5,
        )
        existing_text = "\n".join(r.content for r in existing)

        return await self._fabric.remember(
            content=content,
            source="business_decision",
            metadata={
                "doc_type": "decision",
                "has_outcome": outcome is not None,
            },
            existing_context=existing_text,
        )

    async def record_lesson(
        self,
        lesson: str,
        category: str,
        source: str,
    ) -> RememberResult:
        """Record a lesson learned.

        Args:
            lesson: The lesson learned
            category: Category (e.g., "process", "technical", "customer", "strategy")
            source: Where the lesson came from (e.g., "retrospective", "incident")

        Returns:
            RememberResult
        """
        content = f"[lesson:{category}] {lesson}\nSource: {source}"

        return await self._fabric.remember(
            content=content,
            source="business_lesson",
            metadata={
                "doc_type": "lesson_learned",
                "category": category,
                "lesson_source": source,
            },
        )

    async def get_customer_history(
        self,
        customer_id: str,
        limit: int = 20,
    ) -> list[FabricResult]:
        """Get interaction history for a customer.

        Args:
            customer_id: Customer identifier
            limit: Max results

        Returns:
            List of FabricResult for this customer
        """
        return await self._fabric.query(
            f"customer {customer_id} interaction",
            limit=limit,
        )

    async def get_decision_context(
        self,
        topic: str,
        limit: int = 10,
    ) -> list[FabricResult]:
        """Get past decisions related to a topic.

        Args:
            topic: Topic to search for
            limit: Max results

        Returns:
            List of FabricResult with past decisions
        """
        return await self._fabric.query(
            f"decision {topic}",
            limit=limit,
        )


__all__ = ["BusinessKnowledgeIngester", "BusinessDocument"]

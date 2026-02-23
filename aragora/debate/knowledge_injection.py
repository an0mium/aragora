"""Knowledge Injection - Completes the Receipt -> KM -> Next Debate flywheel.

Queries the Knowledge Mound for relevant past debate receipts and injects
them as context into new debate prompts. This closes the loop:

    Debate -> Receipt -> KM (persist) -> KM (query) -> Next Debate prompt

Without this module, receipts are persisted to the Knowledge Mound by
PostDebateCoordinator._step_persist_receipt() but never retrieved for
new debates.  The knowledge flywheel was broken: data went in but never
came back out.

Usage:
    from aragora.debate.knowledge_injection import (
        DebateKnowledgeInjector,
        KnowledgeInjectionConfig,
    )

    injector = DebateKnowledgeInjector()
    enriched_prompt = await injector.inject_into_prompt(
        base_prompt="You are debating...",
        task="Design a rate limiter",
        domain="engineering",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PastDebateKnowledge:
    """Knowledge extracted from a past debate receipt stored in the KM."""

    debate_id: str
    task: str
    final_answer: str
    confidence: float
    consensus_reached: bool
    relevance_score: float
    key_insights: list[str] = field(default_factory=list)
    dissenting_views: list[str] = field(default_factory=list)


@dataclass
class KnowledgeInjectionConfig:
    """Configuration for injecting KM knowledge into debates."""

    enable_injection: bool = True
    max_relevant_receipts: int = 3
    min_relevance_score: float = 0.3
    include_confidence: bool = True
    include_dissenting_views: bool = True
    max_context_tokens: int = 500


class DebateKnowledgeInjector:
    """Queries KM for relevant past debate receipts and injects them as context.

    Completes the knowledge flywheel:
    Debate -> Receipt -> KM -> Next Debate prompt context
    """

    def __init__(self, config: KnowledgeInjectionConfig | None = None) -> None:
        self.config = config or KnowledgeInjectionConfig()

    async def query_relevant_knowledge(
        self,
        task: str,
        domain: str | None = None,
    ) -> list[PastDebateKnowledge]:
        """Query KM for past debate receipts relevant to the current task.

        Uses the Knowledge Mound's semantic search to find past debates
        on related topics, then extracts useful context.

        Args:
            task: The current debate task/question.
            domain: Optional domain hint to narrow search (e.g. "engineering").

        Returns:
            List of PastDebateKnowledge items sorted by relevance,
            limited by max_relevant_receipts.  Returns empty list when
            KM is unavailable or has no matching data.
        """
        if not self.config.enable_injection:
            return []

        try:
            from aragora.knowledge.mound import get_knowledge_mound
            from aragora.knowledge.unified.types import (
                KnowledgeSource,
                QueryFilters,
            )
        except ImportError:
            logger.debug("Knowledge Mound not available, skipping injection")
            return []

        try:
            mound = get_knowledge_mound()
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Failed to get Knowledge Mound: %s", exc)
            return []

        # Build filters to narrow to receipt-derived knowledge items
        filters = QueryFilters(
            sources=[KnowledgeSource.DEBATE],
            tags=["decision_receipt"],
        )

        try:
            query_result = await mound.query(
                query=task,
                filters=filters,
                limit=self.config.max_relevant_receipts * 2,  # over-fetch then filter
            )
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            logger.warning("KM query failed: %s", exc)
            return []

        items = query_result.items if hasattr(query_result, "items") else []
        if not items:
            return []

        knowledge_list: list[PastDebateKnowledge] = []
        for idx, item in enumerate(items):
            knowledge = self._extract_knowledge(item, idx, len(items), domain)
            if (
                knowledge is not None
                and knowledge.relevance_score >= self.config.min_relevance_score
            ):
                knowledge_list.append(knowledge)

        # Sort by relevance descending
        knowledge_list.sort(key=lambda k: k.relevance_score, reverse=True)

        # Enforce limit
        return knowledge_list[: self.config.max_relevant_receipts]

    def format_for_injection(self, knowledge: list[PastDebateKnowledge]) -> str:
        """Format retrieved knowledge as prompt context.

        Returns a markdown-formatted string suitable for prepending to
        debate prompts.  Returns empty string when no knowledge provided.
        """
        if not knowledge:
            return ""

        sections: list[str] = ["## Relevant Past Decisions\n"]

        for item in knowledge:
            section = self._format_single(item)
            sections.append(section)

        full_text = "\n".join(sections)

        # Approximate token budget enforcement (1 token ~ 4 chars)
        max_chars = self.config.max_context_tokens * 4
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars].rsplit("\n", 1)[0] + "\n..."

        return full_text

    async def inject_into_prompt(
        self,
        base_prompt: str,
        task: str,
        domain: str | None = None,
    ) -> str:
        """Full pipeline: query KM -> format -> inject into prompt.

        Args:
            base_prompt: The original debate prompt to augment.
            task: The current debate task/question.
            domain: Optional domain hint for filtering.

        Returns:
            The enriched prompt with past knowledge appended, or the
            original base_prompt if no relevant knowledge is found.
        """
        if not self.config.enable_injection:
            return base_prompt

        knowledge = await self.query_relevant_knowledge(task, domain)
        if not knowledge:
            return base_prompt

        context = self.format_for_injection(knowledge)
        if not context:
            return base_prompt

        return f"{base_prompt}\n\n{context}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_knowledge(
        self,
        item: Any,
        index: int,
        total: int,
        domain: str | None,
    ) -> PastDebateKnowledge | None:
        """Extract structured knowledge from a KnowledgeItem.

        The relevance_score is derived from the item's position in the
        result list (which is ordered by the KM's own scoring) and
        optionally boosted when the domain matches.
        """
        metadata: dict[str, Any] = getattr(item, "metadata", {}) or {}
        content: str = getattr(item, "content", "") or ""

        debate_id = metadata.get("receipt_id", metadata.get("debate_id", ""))
        task = metadata.get("task", "")
        if not task:
            # Fall back: try to extract from content
            for line in content.split("\n"):
                if line.startswith("Input:"):
                    task = line[len("Input:") :].strip()[:200]
                    break

        final_answer = metadata.get("final_answer", "")
        if not final_answer:
            verdict = metadata.get("verdict", "")
            if verdict:
                final_answer = f"Verdict: {verdict}"

        confidence = float(metadata.get("confidence", 0.5))
        consensus_reached = bool(metadata.get("consensus_reached", False))

        # Position-based relevance: first result gets 1.0, last gets lower bound
        if total > 1:
            relevance = 1.0 - (index * 0.6 / (total - 1))
        else:
            relevance = 1.0

        # Boost if domain matches
        tags: list[str] = metadata.get("tags", [])
        if domain and any(domain.lower() in t.lower() for t in tags):
            relevance = min(1.0, relevance + 0.1)

        key_insights: list[str] = metadata.get("key_insights", [])
        dissenting_views: list[str] = metadata.get("dissenting_views", [])

        return PastDebateKnowledge(
            debate_id=debate_id,
            task=task,
            final_answer=final_answer,
            confidence=confidence,
            consensus_reached=consensus_reached,
            relevance_score=round(relevance, 3),
            key_insights=key_insights,
            dissenting_views=dissenting_views,
        )

    def _format_single(self, item: PastDebateKnowledge) -> str:
        """Format a single PastDebateKnowledge item as markdown."""
        parts: list[str] = []

        header = f"**{item.task or item.debate_id}**"
        if self.config.include_confidence:
            header += f" (confidence: {item.confidence:.2f})"
        parts.append(header)

        if item.final_answer:
            parts.append(f"- Decision: {item.final_answer}")

        if item.key_insights:
            for insight in item.key_insights:
                parts.append(f"- Key insight: {insight}")

        if self.config.include_dissenting_views and item.dissenting_views:
            for view in item.dissenting_views:
                parts.append(f"- Dissenting view: {view}")

        return "\n".join(parts) + "\n"


__all__ = [
    "DebateKnowledgeInjector",
    "KnowledgeInjectionConfig",
    "PastDebateKnowledge",
]

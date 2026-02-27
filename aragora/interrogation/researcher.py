"""Gather context from multiple sources for each interrogation dimension."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aragora.interrogation.decomposer import Dimension

logger = logging.getLogger(__name__)


class ResearchSource(Enum):
    """Sources that the researcher can query for context."""

    KNOWLEDGE_MOUND = "knowledge_mound"
    OBSIDIAN = "obsidian"
    CODEBASE = "codebase"
    WEB = "web"


@dataclass
class Finding:
    """A single piece of research context."""

    source: ResearchSource
    content: str
    relevance: float  # 0.0-1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Research findings grouped by dimension."""

    findings: dict[str, list[Finding]] = field(default_factory=dict)

    @property
    def total_findings(self) -> int:
        return sum(len(f) for f in self.findings.values())

    def for_dimension(self, dimension_name: str) -> list[Finding]:
        return self.findings.get(dimension_name, [])


class InterrogationResearcher:
    """Gathers context from KnowledgeMound, Obsidian, codebase, and web."""

    def __init__(
        self,
        knowledge_mound: Any | None = None,
        obsidian: Any | None = None,
    ) -> None:
        self._km = knowledge_mound
        self._obsidian = obsidian

    async def research(
        self,
        dimensions: list[Dimension],
        sources: list[str] | None = None,
    ) -> ResearchResult:
        """Research each dimension using the specified sources.

        Args:
            dimensions: Dimensions to research (from decomposer).
            sources: Source names to query (e.g. ["knowledge_mound", "obsidian"]).
                     Empty or None means no external queries.

        Returns:
            ResearchResult with findings grouped by dimension name.
        """
        result = ResearchResult()
        if not dimensions:
            return result

        enabled = set(sources or [])

        for dim in dimensions:
            dim_findings: list[Finding] = []

            if "knowledge_mound" in enabled and self._km:
                dim_findings.extend(await self._query_km(dim))

            if "obsidian" in enabled and self._obsidian:
                dim_findings.extend(await self._query_obsidian(dim))

            result.findings[dim.name] = dim_findings

        return result

    async def _query_km(self, dim: Dimension) -> list[Finding]:
        """Query KnowledgeMound for prior debate context on a dimension."""
        try:
            query_result = await self._km.query(
                query=f"{dim.name}: {dim.description}",
                limit=5,
            )
            return [
                Finding(
                    source=ResearchSource.KNOWLEDGE_MOUND,
                    content=item.content if hasattr(item, "content") else str(item),
                    relevance=0.7,
                    metadata=getattr(item, "metadata", {}),
                )
                for item in (
                    query_result.items if hasattr(query_result, "items") else []
                )
            ]
        except Exception:
            logger.warning(
                "KnowledgeMound query failed for dimension %s",
                dim.name,
                exc_info=True,
            )
            return []

    async def _query_obsidian(self, dim: Dimension) -> list[Finding]:
        """Query Obsidian vault for user notes relevant to a dimension."""
        try:
            notes = await self._obsidian.search(
                query=dim.name,
                limit=5,
            )
            return [
                Finding(
                    source=ResearchSource.OBSIDIAN,
                    content=note.content if hasattr(note, "content") else str(note),
                    relevance=0.6,
                    metadata=getattr(note, "metadata", {}),
                )
                for note in notes
            ]
        except Exception:
            logger.warning(
                "Obsidian query failed for dimension %s",
                dim.name,
                exc_info=True,
            )
            return []

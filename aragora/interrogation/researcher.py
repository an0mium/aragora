"""Unified Researcher Component."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ResearchSource(str, Enum):
    """Legacy source enum used by interrogation questioning/tests."""

    KNOWLEDGE_MOUND = "knowledge_mound"
    OBSIDIAN = "obsidian"
    CODEBASE = "codebase"
    WEB = "web"


@dataclass
class Finding:
    """Legacy finding object grouped by prompt dimension."""

    source: ResearchSource
    content: str
    relevance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Legacy research container grouped by dimension name."""

    findings: dict[str, list[Finding]] = field(default_factory=dict)

    def for_dimension(self, dimension_name: str) -> list[Finding]:
        return list(self.findings.get(dimension_name, []))

    def add_finding(self, dimension_name: str, finding: Finding) -> None:
        self.findings.setdefault(dimension_name, []).append(finding)


@dataclass
class SourceResult:
    source: str
    content: str
    relevance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchContext:
    query: str
    results: list[SourceResult] = field(default_factory=list)
    sources_queried: list[str] = field(default_factory=list)
    sources_failed: list[str] = field(default_factory=list)

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    @property
    def top_results(self) -> list[SourceResult]:
        return sorted(self.results, key=lambda r: r.relevance, reverse=True)

    def by_source(self, source: str) -> list[SourceResult]:
        return [r for r in self.results if r.source == source]

    def summary(self, max_chars: int = 2000) -> str:
        parts: list[str] = []
        for result in self.top_results:
            if sum(len(p) for p in parts) + len(result.content) > max_chars:
                break
            parts.append(f"[{result.source}] {result.content}")
        return "\n\n".join(parts) if parts else "No research findings."


class UnifiedResearcher:
    def __init__(
        self,
        knowledge_mound: Any | None = None,
        obsidian_adapter: Any | None = None,
        codebase_analyzer: Any | None = None,
        web_searcher: Any | None = None,
    ) -> None:
        self._km = knowledge_mound
        self._obsidian = obsidian_adapter
        self._codebase = codebase_analyzer
        self._web = web_searcher

    async def research(
        self,
        query: str,
        sources: list[str] | None = None,
        max_results_per_source: int = 5,
    ) -> ResearchContext:
        ctx = ResearchContext(query=query)
        target_sources = sources or ["km", "obsidian", "codebase", "web"]
        for source in target_sources:
            try:
                results = await self._query_source(source, query, max_results_per_source)
                ctx.results.extend(results)
                ctx.sources_queried.append(source)
            except Exception:
                logger.warning("Research source %s failed for query: %s", source, query)
                ctx.sources_failed.append(source)
        return ctx

    async def _query_source(self, source: str, query: str, max_results: int) -> list[SourceResult]:
        if source == "km":
            return await self._query_km(query, max_results)
        elif source == "obsidian":
            return await self._query_obsidian(query, max_results)
        elif source == "codebase":
            return await self._query_codebase(query, max_results)
        elif source == "web":
            return await self._query_web(query, max_results)
        return []

    async def _query_km(self, query: str, max_results: int) -> list[SourceResult]:
        if self._km is None:
            return []
        results = self._km.query(query, limit=max_results)
        if not results:
            return []
        return [
            SourceResult(
                source="km",
                content=str(r.get("content", r.get("text", str(r)))),
                relevance=float(r.get("relevance", r.get("score", 0.5))),
                metadata={"type": r.get("type", "unknown")},
            )
            for r in (results if isinstance(results, list) else [results])
        ][:max_results]

    async def _query_obsidian(self, query: str, max_results: int) -> list[SourceResult]:
        if self._obsidian is None:
            return []
        results = self._obsidian.search(query, limit=max_results)
        if not results:
            return []
        return [
            SourceResult(
                source="obsidian",
                content=str(r.get("content", str(r))),
                relevance=float(r.get("relevance", 0.5)),
                metadata={"title": r.get("title", ""), "path": r.get("path", "")},
            )
            for r in (results if isinstance(results, list) else [results])
        ][:max_results]

    async def _query_codebase(self, query: str, max_results: int) -> list[SourceResult]:
        if self._codebase is None:
            return []
        results = self._codebase.analyze(query, limit=max_results)
        if not results:
            return []
        return [
            SourceResult(
                source="codebase",
                content=str(r.get("content", str(r))),
                relevance=float(r.get("relevance", 0.5)),
                metadata={"file": r.get("file", ""), "type": r.get("type", "pattern")},
            )
            for r in (results if isinstance(results, list) else [results])
        ][:max_results]

    async def _query_web(self, query: str, max_results: int) -> list[SourceResult]:
        if self._web is None:
            return []
        results = self._web.search(query, limit=max_results)
        if not results:
            return []
        return [
            SourceResult(
                source="web",
                content=str(r.get("content", r.get("snippet", str(r)))),
                relevance=float(r.get("relevance", 0.5)),
                metadata={"url": r.get("url", ""), "title": r.get("title", "")},
            )
            for r in (results if isinstance(results, list) else [results])
        ][:max_results]

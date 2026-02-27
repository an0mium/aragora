"""MultiSourceResearcher: fans out research to enabled sources."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aragora.prompt_engine.types import (
    RefinedIntent,
    ResearchReport,
    ResearchSource,
)

logger = logging.getLogger(__name__)


class MultiSourceResearcher:
    """Fans out research queries to multiple sources via asyncio.gather()."""

    async def research(
        self,
        refined_intent: RefinedIntent,
        sources: list[ResearchSource],
    ) -> ResearchReport:
        report = ResearchReport(
            km_precedents=[],
            codebase_context=[],
            obsidian_notes=[],
            web_results=[],
            sources_used=[],
        )
        if not sources:
            return report

        tasks = {}
        for source in sources:
            if source == ResearchSource.KNOWLEDGE_MOUND:
                tasks[source] = self._research_km(refined_intent)
            elif source == ResearchSource.CODEBASE:
                tasks[source] = self._research_codebase(refined_intent)
            elif source == ResearchSource.OBSIDIAN:
                tasks[source] = self._research_obsidian(refined_intent)
            elif source == ResearchSource.WEB:
                tasks[source] = self._research_web(refined_intent)

        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True,
        )

        for source, result in zip(tasks.keys(), results):
            if isinstance(result, BaseException):
                logger.warning("Research source %s failed: %s", source.value, result)
                continue
            report.sources_used.append(source)
            if source == ResearchSource.KNOWLEDGE_MOUND:
                report.km_precedents = result
            elif source == ResearchSource.CODEBASE:
                report.codebase_context = result
            elif source == ResearchSource.OBSIDIAN:
                report.obsidian_notes = result
            elif source == ResearchSource.WEB:
                report.web_results = result

        return report

    async def _research_km(self, intent: RefinedIntent) -> list[dict[str, Any]]:
        try:
            from aragora.pipeline.km_bridge import PipelineKMBridge

            bridge = PipelineKMBridge()
            if not bridge.available:
                return []
            return bridge.query_all_adapter_precedents(
                intent.intent.raw_prompt,
                limit=5,
            ).get("debates", [])
        except ImportError:
            logger.debug("PipelineKMBridge not available")
            return []

    async def _research_codebase(
        self,
        intent: RefinedIntent,
    ) -> list[dict[str, Any]]:
        return []

    async def _research_obsidian(
        self,
        intent: RefinedIntent,
    ) -> list[dict[str, Any]]:
        return []

    async def _research_web(
        self,
        intent: RefinedIntent,
    ) -> list[dict[str, Any]]:
        return []

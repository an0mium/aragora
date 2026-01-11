"""
Context gathering for debate research and evidence collection.

Extracts research-related context gathering from the Arena class
to improve maintainability and allow independent testing.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ContextGatherer:
    """
    Gathers context from multiple sources for debate grounding.

    Sources include:
    - Aragora project documentation (for self-referential debates)
    - Web search via EvidenceCollector
    - GitHub repositories
    - Local documentation
    - Pulse/trending topics from social platforms

    Usage:
        gatherer = ContextGatherer(evidence_store_callback=store_evidence)
        context = await gatherer.gather_all(task="Discuss AI safety")
    """

    def __init__(
        self,
        evidence_store_callback: Optional[Callable[..., Any]] = None,
        prompt_builder: Optional[Any] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize the context gatherer.

        Args:
            evidence_store_callback: Optional callback to store evidence snippets.
                                    Signature: (snippets: list, task: str) -> None
            prompt_builder: Optional PromptBuilder to receive evidence pack.
            project_root: Optional project root path for documentation lookup.
                         Defaults to detecting from this file's location.
        """
        self._evidence_store_callback = evidence_store_callback
        self._prompt_builder = prompt_builder
        self._project_root = project_root or Path(__file__).parent.parent.parent

        # Cache for evidence pack (for grounding verdict with citations)
        self._research_evidence_pack = None

        # Cache for research context
        self._research_context_cache: Optional[str] = None

    @property
    def evidence_pack(self) -> Optional[Any]:
        """Get the cached evidence pack from the last research call."""
        return self._research_evidence_pack

    def set_prompt_builder(self, prompt_builder: Any) -> None:
        """Set or update the prompt builder reference."""
        self._prompt_builder = prompt_builder

    async def gather_all(self, task: str) -> str:
        """
        Perform multi-source research and return formatted context.

        Gathers context from:
        - Aragora documentation (if task is Aragora-related)
        - Evidence connectors (web, GitHub, local docs)
        - Pulse/trending topics

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted context string, or "No research context available."
        """
        if self._research_context_cache:
            return self._research_context_cache

        context_parts = []

        # Gather context from multiple sources
        aragora_ctx = await self.gather_aragora_context(task)
        if aragora_ctx:
            context_parts.append(aragora_ctx)

        evidence_ctx = await self.gather_evidence_context(task)
        if evidence_ctx:
            context_parts.append(evidence_ctx)

        trending_ctx = await self.gather_trending_context()
        if trending_ctx:
            context_parts.append(trending_ctx)

        if context_parts:
            self._research_context_cache = "\n\n".join(context_parts)
            return self._research_context_cache
        else:
            return "No research context available."

    async def gather_aragora_context(self, task: str) -> Optional[str]:
        """
        Gather Aragora-specific documentation context if task is relevant.

        Only activates for tasks mentioning Aragora, multi-agent debates,
        decision stress-tests, nomic loop, or the debate framework.

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted documentation context, or None if not relevant.
        """
        task_lower = task.lower()
        is_aragora_topic = any(
            kw in task_lower
            for kw in [
                "aragora",
                "multi-agent debate",
                "decision stress-test",
                "ai red team",
                "adversarial validation",
                "gauntlet",
                "nomic loop",
                "debate framework",
            ]
        )

        if not is_aragora_topic:
            return None

        try:
            docs_dir = self._project_root / "docs"
            aragora_context_parts: list[str] = []
            loop = asyncio.get_running_loop()

            def _read_file_sync(path: Path, limit: int) -> str | None:
                try:
                    if path.exists():
                        return path.read_text()[:limit]
                except (OSError, UnicodeDecodeError) as e:
                    logger.debug(f"Failed to read file {path}: {e}")
                return None

            # Read key documentation files
            key_docs = ["FEATURES.md", "ARCHITECTURE.md", "QUICKSTART.md", "STATUS.md"]
            for doc_name in key_docs:
                doc_path = docs_dir / doc_name
                content = await loop.run_in_executor(
                    None, lambda p=doc_path: _read_file_sync(p, 3000)  # type: ignore[misc]
                )
                if content:
                    aragora_context_parts.append(f"### {doc_name}\n{content}")

            # Also include CLAUDE.md for project overview
            claude_md = self._project_root / "CLAUDE.md"
            content = await loop.run_in_executor(
                None, lambda: _read_file_sync(claude_md, 2000)
            )
            if content:
                aragora_context_parts.insert(0, f"### Project Overview (CLAUDE.md)\n{content}")

            if aragora_context_parts:
                logger.info("Injected Aragora project documentation context")
                return (
                    "## ARAGORA PROJECT CONTEXT\n"
                    "The following is internal documentation about the Aragora project:\n\n"
                    + "\n\n---\n\n".join(aragora_context_parts[:4])
                )

        except Exception as e:
            logger.warning(f"Failed to load Aragora context: {e}")

        return None

    async def gather_evidence_context(self, task: str) -> Optional[str]:
        """
        Gather evidence from web, GitHub, and local docs connectors.

        Uses EvidenceCollector with available connectors:
        - WebConnector: DuckDuckGo search (if duckduckgo_search installed)
        - GitHubConnector: Code/docs from GitHub (if GITHUB_TOKEN set)
        - LocalDocsConnector: Local documentation files

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted evidence context, or None if unavailable.
        """
        try:
            from aragora.evidence.collector import EvidenceCollector

            collector = EvidenceCollector()
            enabled_connectors = []

            # Add web connector if available
            try:
                from aragora.connectors.web import WebConnector, DDGS_AVAILABLE
                if DDGS_AVAILABLE:
                    collector.add_connector("web", WebConnector())
                    enabled_connectors.append("web")
            except ImportError:
                pass

            # Add GitHub connector if available
            try:
                from aragora.connectors.github import GitHubConnector
                import os
                if os.environ.get("GITHUB_TOKEN"):
                    collector.add_connector("github", GitHubConnector())
                    enabled_connectors.append("github")
            except ImportError:
                pass

            # Add local docs connector
            try:
                from aragora.connectors.local_docs import LocalDocsConnector

                collector.add_connector("local_docs", LocalDocsConnector(
                    root_path=str(self._project_root / "docs"),
                    file_types="docs"
                ))
                enabled_connectors.append("local_docs")
            except ImportError:
                pass

            if not enabled_connectors:
                return None

            evidence_pack = await collector.collect_evidence(
                task, enabled_connectors=enabled_connectors
            )

            if evidence_pack.snippets:
                self._research_evidence_pack = evidence_pack  # type: ignore[assignment]

                # Update prompt builder if available
                if self._prompt_builder:
                    self._prompt_builder.set_evidence_pack(evidence_pack)

                # Store evidence via callback if provided
                if self._evidence_store_callback and callable(self._evidence_store_callback):
                    self._evidence_store_callback(evidence_pack.snippets, task)

                return f"## EVIDENCE CONTEXT\n{evidence_pack.to_context_string()}"

        except Exception as e:
            logger.warning(f"Evidence collection failed: {e}")

        return None

    async def gather_trending_context(self) -> Optional[str]:
        """
        Gather pulse/trending context from social platforms.

        Uses PulseManager with available ingestors:
        - Twitter
        - Hacker News
        - Reddit

        Returns:
            Formatted trending topics context, or None if unavailable.
        """
        try:
            from aragora.pulse.ingestor import (
                PulseManager,
                TwitterIngestor,
                HackerNewsIngestor,
                RedditIngestor,
            )

            manager = PulseManager()
            manager.add_ingestor("twitter", TwitterIngestor())
            manager.add_ingestor("hackernews", HackerNewsIngestor())
            manager.add_ingestor("reddit", RedditIngestor())

            topics = await manager.get_trending_topics(limit_per_platform=3)

            if topics:
                trending_context = "## TRENDING CONTEXT\nCurrent trending topics that may be relevant:\n"
                for t in topics[:5]:
                    trending_context += f"- {t.topic} ({t.platform}, {t.volume:,} engagement, {t.category})\n"
                return trending_context

        except Exception as e:
            logger.debug(f"Pulse context unavailable: {e}")

        return None

    def clear_cache(self) -> None:
        """Clear all cached context."""
        self._research_context_cache = None
        self._research_evidence_pack = None

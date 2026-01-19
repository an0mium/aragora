"""
Context gathering for debate research and evidence collection.

Extracts research-related context gathering from the Arena class
to improve maintainability and allow independent testing.

Timeouts:
    CONTEXT_GATHER_TIMEOUT: Overall timeout for gather_all (default: 10s)
    EVIDENCE_TIMEOUT: Timeout for evidence collection (default: 5s)
    TRENDING_TIMEOUT: Timeout for trending topics (default: 3s)

RLM Integration:
    When enable_rlm_compression is True and a compressor is available,
    large documents are hierarchically compressed instead of truncated.
    This preserves semantic content while fitting within token budgets.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.rlm import HierarchicalCompressor

# Check for RLM availability (prefer AragoraRLM for TRUE RLM support)
try:
    from aragora.rlm.bridge import AragoraRLM, HAS_OFFICIAL_RLM
    from aragora.rlm import HierarchicalCompressor as _HierarchicalCompressor
    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HAS_OFFICIAL_RLM = False
    AragoraRLM = None  # type: ignore[misc,assignment]
    _HierarchicalCompressor = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

# Configurable timeouts (in seconds)
# Increased timeouts to allow Claude web search to complete
CONTEXT_GATHER_TIMEOUT = float(os.getenv("ARAGORA_CONTEXT_TIMEOUT", "150.0"))
CLAUDE_SEARCH_TIMEOUT = float(os.getenv("ARAGORA_CLAUDE_SEARCH_TIMEOUT", "120.0"))
EVIDENCE_TIMEOUT = float(os.getenv("ARAGORA_EVIDENCE_TIMEOUT", "30.0"))
TRENDING_TIMEOUT = float(os.getenv("ARAGORA_TRENDING_TIMEOUT", "5.0"))


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
        enable_rlm_compression: bool = True,
        rlm_compressor: Optional["HierarchicalCompressor"] = None,
        rlm_compression_threshold: int = 3000,  # Chars above which to use RLM
    ):
        """
        Initialize the context gatherer.

        Args:
            evidence_store_callback: Optional callback to store evidence snippets.
                                    Signature: (snippets: list, task: str) -> None
            prompt_builder: Optional PromptBuilder to receive evidence pack.
            project_root: Optional project root path for documentation lookup.
                         Defaults to detecting from this file's location.
            enable_rlm_compression: Whether to use RLM for large document compression.
            rlm_compressor: Optional pre-configured HierarchicalCompressor.
            rlm_compression_threshold: Char count above which to apply RLM compression.
        """
        self._evidence_store_callback = evidence_store_callback
        self._prompt_builder = prompt_builder
        self._project_root = project_root or Path(__file__).parent.parent.parent

        # Cache for evidence pack (for grounding verdict with citations)
        self._research_evidence_pack: Optional[Any] = None

        # Cache for research context
        self._research_context_cache: Optional[str] = None

        # Cache for continuum memory context
        self._continuum_context_cache: Optional[str] = None

        # RLM configuration - prefer AragoraRLM for TRUE RLM (REPL-based) support
        self._enable_rlm = enable_rlm_compression and HAS_RLM
        self._rlm_compressor = rlm_compressor
        self._aragora_rlm: Optional[Any] = None
        self._rlm_threshold = rlm_compression_threshold

        if self._enable_rlm:
            # Prefer AragoraRLM which routes to TRUE RLM when available
            if AragoraRLM is not None:
                try:
                    self._aragora_rlm = AragoraRLM()
                    if HAS_OFFICIAL_RLM:
                        logger.info(
                            "[rlm] ContextGatherer: TRUE RLM enabled "
                            "(REPL-based, model writes code to examine context)"
                        )
                    else:
                        logger.info(
                            "[rlm] ContextGatherer: AragoraRLM enabled "
                            "(will use compression fallback since official RLM not installed)"
                        )
                except Exception as e:
                    logger.warning(f"[rlm] Failed to initialize AragoraRLM: {e}")

            # Fallback: direct HierarchicalCompressor (compression-only)
            if not self._rlm_compressor and _HierarchicalCompressor:
                self._rlm_compressor = _HierarchicalCompressor()
                logger.debug("[rlm] ContextGatherer: HierarchicalCompressor fallback enabled")

    @property
    def evidence_pack(self) -> Optional[Any]:
        """Get the cached evidence pack from the last research call."""
        return self._research_evidence_pack

    def set_prompt_builder(self, prompt_builder: Any) -> None:
        """Set or update the prompt builder reference."""
        self._prompt_builder = prompt_builder

    async def gather_all(self, task: str, timeout: Optional[float] = None) -> str:
        """
        Perform multi-source research and return formatted context.

        Gathers context from:
        - Claude's web search (primary - best quality, uses Opus 4.5)
        - Aragora documentation (if task is Aragora-related)
        - Evidence connectors (web, GitHub, local docs) - fallback
        - Pulse/trending topics

        All sub-operations have individual timeouts to prevent blocking.
        Returns partial results if some sources timeout.

        Args:
            task: The debate topic/task description.
            timeout: Overall timeout in seconds (default: CONTEXT_GATHER_TIMEOUT)

        Returns:
            Formatted context string, or "No research context available."
        """
        if self._research_context_cache:
            return self._research_context_cache

        timeout = timeout or CONTEXT_GATHER_TIMEOUT
        context_parts = []

        async def _gather_with_timeout():
            nonlocal context_parts

            # 1. Primary: Claude's web search (best quality research)
            claude_ctx = await self._gather_claude_web_search(task)
            if claude_ctx:
                context_parts.append(claude_ctx)

            # 2. Gather Aragora context (local files, fast)
            aragora_ctx = await self.gather_aragora_context(task)
            if aragora_ctx:
                context_parts.append(aragora_ctx)

            # 3. ALWAYS gather trending context for real-time relevance
            # This provides current context even when Claude search succeeds
            trending_task = asyncio.create_task(self._gather_trending_with_timeout())

            # 4. Gather additional evidence in parallel (fallback if Claude search weak)
            if not claude_ctx or len(claude_ctx) < 500:
                evidence_task = asyncio.create_task(self._gather_evidence_with_timeout(task))
                results = await asyncio.gather(evidence_task, trending_task, return_exceptions=True)
            else:
                # Still wait for trending even if Claude search succeeded
                results = await asyncio.gather(trending_task, return_exceptions=True)

            for result in results:
                if isinstance(result, str) and result:
                    context_parts.append(result)
                elif isinstance(result, asyncio.TimeoutError):
                    logger.warning("Context gathering subtask timed out")
                elif isinstance(result, Exception):
                    logger.debug(f"Context gathering subtask failed: {result}")

        try:
            await asyncio.wait_for(_gather_with_timeout(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Context gathering timed out after {timeout}s, using partial results")

        if context_parts:
            self._research_context_cache = "\n\n".join(context_parts)
            return self._research_context_cache
        else:
            return "No research context available."

    async def _gather_claude_web_search(self, task: str) -> Optional[str]:
        """
        Perform web search using Claude's built-in web_search tool.

        This is the primary research method - uses Claude Opus 4.5 with
        web search capability to provide high-quality, current information.

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted research context, or None if search fails.
        """
        try:
            from aragora.server.research_phase import research_for_debate

            logger.info("[research] Starting Claude web search for debate context...")

            result = await asyncio.wait_for(
                research_for_debate(task),
                timeout=CLAUDE_SEARCH_TIMEOUT,
            )

            if result:
                logger.info(f"[research] Claude web search complete: {len(result)} chars")
                return result
            else:
                logger.info("[research] Claude web search returned no results")
                return None

        except asyncio.TimeoutError:
            logger.warning(f"[research] Claude web search timed out after {CLAUDE_SEARCH_TIMEOUT}s")
            return None
        except ImportError:
            logger.debug("[research] research_phase module not available")
            return None
        except Exception as e:
            logger.warning(f"[research] Claude web search failed: {e}")
            return None

    async def _gather_evidence_with_timeout(self, task: str) -> Optional[str]:
        """Gather evidence with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.gather_evidence_context(task), timeout=EVIDENCE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Evidence collection timed out after {EVIDENCE_TIMEOUT}s")
            return None

    async def _gather_trending_with_timeout(self) -> Optional[str]:
        """Gather trending context with timeout protection."""
        try:
            return await asyncio.wait_for(self.gather_trending_context(), timeout=TRENDING_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Trending context timed out after {TRENDING_TIMEOUT}s")
            return None

    async def gather_aragora_context(self, task: str) -> Optional[str]:
        """
        Gather Aragora-specific documentation context if task is relevant.

        Only activates for tasks mentioning Aragora, multi-agent debates,
        decision stress-tests, nomic loop, or the debate framework.

        Uses RLM compression for large documents to preserve semantic content
        instead of simple truncation.

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

            def _read_file_sync(path: Path) -> str | None:
                """Read full file content without truncation."""
                try:
                    if path.exists():
                        return path.read_text()
                except (OSError, UnicodeDecodeError) as e:
                    logger.debug(f"Failed to read file {path}: {e}")
                return None

            # Read key documentation files (full content, RLM will compress)
            key_docs = ["FEATURES.md", "ARCHITECTURE.md", "QUICKSTART.md", "STATUS.md"]
            for doc_name in key_docs:
                doc_path = docs_dir / doc_name
                content = await loop.run_in_executor(
                    None,
                    lambda p=doc_path: _read_file_sync(p),  # type: ignore[misc]
                )
                if content:
                    # Use RLM to compress if content is large
                    compressed = await self._compress_with_rlm(
                        content,
                        source_type="documentation",
                        max_chars=3000,
                    )
                    aragora_context_parts.append(f"### {doc_name}\n{compressed}")

            # Also include CLAUDE.md for project overview
            claude_md = self._project_root / "CLAUDE.md"
            content = await loop.run_in_executor(None, lambda: _read_file_sync(claude_md))
            if content:
                # Compress CLAUDE.md with RLM if large
                compressed = await self._compress_with_rlm(
                    content,
                    source_type="documentation",
                    max_chars=2000,
                )
                aragora_context_parts.insert(0, f"### Project Overview (CLAUDE.md)\n{compressed}")

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
                from aragora.connectors.web import DDGS_AVAILABLE, WebConnector

                if DDGS_AVAILABLE:
                    collector.add_connector("web", WebConnector())
                    enabled_connectors.append("web")
            except ImportError:
                pass

            # Add GitHub connector if available
            try:
                import os

                from aragora.connectors.github import GitHubConnector

                if os.environ.get("GITHUB_TOKEN"):
                    collector.add_connector("github", GitHubConnector())
                    enabled_connectors.append("github")
            except ImportError:
                pass

            # Add local docs connector
            try:
                from aragora.connectors.local_docs import LocalDocsConnector

                collector.add_connector(
                    "local_docs",
                    LocalDocsConnector(
                        root_path=str(self._project_root / "docs"), file_types="docs"
                    ),
                )
                enabled_connectors.append("local_docs")
            except ImportError:
                pass

            if not enabled_connectors:
                return None

            evidence_pack = await collector.collect_evidence(
                task, enabled_connectors=enabled_connectors
            )

            if evidence_pack.snippets:
                self._research_evidence_pack = evidence_pack

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
                GitHubTrendingIngestor,
                GoogleTrendsIngestor,
                HackerNewsIngestor,
                PulseManager,
                RedditIngestor,
            )

            manager = PulseManager()
            # Free, no-auth sources for real trending data:
            manager.add_ingestor("google", GoogleTrendsIngestor())  # Real Google Trends
            manager.add_ingestor("hackernews", HackerNewsIngestor())  # Real HN front page
            manager.add_ingestor("reddit", RedditIngestor())  # Real Reddit hot posts
            manager.add_ingestor("github", GitHubTrendingIngestor())  # Real GitHub trending

            topics = await manager.get_trending_topics(limit_per_platform=3)

            if topics:
                trending_context = (
                    "## TRENDING CONTEXT\nCurrent trending topics that may be relevant:\n"
                )
                for t in topics[:5]:
                    trending_context += (
                        f"- {t.topic} ({t.platform}, {t.volume:,} engagement, {t.category})\n"
                    )
                return trending_context

        except Exception as e:
            logger.debug(f"Pulse context unavailable: {e}")

        return None

    def clear_cache(self) -> None:
        """Clear all cached context."""
        self._research_context_cache = None
        self._research_evidence_pack = None
        self._continuum_context_cache = None

    async def _compress_with_rlm(
        self,
        content: str,
        source_type: str = "documentation",
        max_chars: int = 3000,
    ) -> str:
        """
        Compress large content using RLM.

        Prioritizes TRUE RLM (REPL-based) via AragoraRLM when available:
        - Model writes code to examine/summarize content
        - Model has agency in deciding how to compress

        Falls back to HierarchicalCompressor (compression-only) if:
        - Official RLM not installed
        - AragoraRLM fails

        Falls back to truncation if all else fails.

        Args:
            content: The content to compress
            source_type: Type of content (for compression hints)
            max_chars: Target character limit

        Returns:
            Compressed or truncated content
        """
        # If content is under threshold, return as-is
        if len(content) <= self._rlm_threshold:
            return content[:max_chars] if len(content) > max_chars else content

        # If RLM is not enabled, use simple truncation
        if not self._enable_rlm:
            return content[:max_chars - 30] + "... [truncated]" if len(content) > max_chars else content

        # PRIMARY: Try AragoraRLM (routes to TRUE RLM if available)
        if self._aragora_rlm:
            try:
                logger.debug(
                    "[rlm] Using AragoraRLM for compression "
                    "(routes to TRUE RLM if available)"
                )
                result = await asyncio.wait_for(
                    self._aragora_rlm.compress_and_query(
                        query=f"Summarize this {source_type} in under {max_chars} characters",
                        content=content,
                        source_type=source_type,
                    ),
                    timeout=15.0,
                )

                if result.answer and len(result.answer) < len(content):
                    approach = "TRUE RLM" if result.used_true_rlm else "compression fallback"
                    logger.debug(
                        f"[rlm] Compressed {len(content)} -> {len(result.answer)} chars "
                        f"({len(result.answer)/len(content)*100:.0f}%) via {approach}"
                    )
                    return result.answer[:max_chars] if len(result.answer) > max_chars else result.answer

            except asyncio.TimeoutError:
                logger.debug("[rlm] AragoraRLM compression timed out")
            except Exception as e:
                logger.debug(f"[rlm] AragoraRLM compression failed: {e}")

        # FALLBACK: Try direct HierarchicalCompressor (compression-only)
        if self._rlm_compressor:
            try:
                logger.debug(
                    "[rlm] Falling back to HierarchicalCompressor "
                    "(compression-only, no TRUE RLM)"
                )
                compression_result = await asyncio.wait_for(
                    self._rlm_compressor.compress(
                        content=content,
                        source_type=source_type,
                        max_levels=2,  # ABSTRACT and SUMMARY for faster compression
                    ),
                    timeout=10.0,
                )

                # Get summary level (or abstract if summary is too long)
                try:
                    from aragora.rlm.types import AbstractionLevel
                    summary = compression_result.context.get_at_level(AbstractionLevel.SUMMARY)
                    if summary and len(summary) > max_chars:
                        summary = compression_result.context.get_at_level(AbstractionLevel.ABSTRACT)
                except (ImportError, AttributeError):
                    summary = None

                if summary and len(summary) < len(content):
                    logger.debug(
                        f"[rlm] Compressed {len(content)} -> {len(summary)} chars "
                        f"({len(summary)/len(content)*100:.0f}%) via HierarchicalCompressor"
                    )
                    return summary[:max_chars] if len(summary) > max_chars else summary

            except asyncio.TimeoutError:
                logger.debug("[rlm] HierarchicalCompressor timed out")
            except Exception as e:
                logger.debug(f"[rlm] HierarchicalCompressor failed: {e}")

        # FINAL FALLBACK: Simple truncation
        logger.debug("[rlm] All RLM approaches failed, using simple truncation")
        return content[:max_chars - 30] + "... [truncated]" if len(content) > max_chars else content

    def get_continuum_context(
        self,
        continuum_memory: Any,
        domain: str,
        task: str,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Retrieve relevant memories from ContinuumMemory for debate context.

        Uses the debate task and domain to query for related past learnings.
        Enhanced with tier-aware retrieval and confidence markers.

        Args:
            continuum_memory: ContinuumMemory instance to query
            domain: The debate domain (e.g., "programming", "ethics")
            task: The debate task description

        Returns:
            Tuple of:
            - Formatted context string
            - List of retrieved memory IDs (for outcome tracking)
            - Dict mapping memory ID to tier (for analytics)
        """
        if hasattr(self, "_continuum_context_cache") and self._continuum_context_cache:
            return self._continuum_context_cache, [], {}

        if not continuum_memory:
            return "", [], {}

        try:
            query = f"{domain}: {task[:200]}"

            # Retrieve memories, prioritizing fast/medium tiers (skip glacial for speed)
            memories = continuum_memory.retrieve(
                query=query,
                limit=5,
                min_importance=0.3,  # Only important memories
            )

            if not memories:
                return "", [], {}

            # Track retrieved memory IDs and tiers for outcome updates and analytics
            retrieved_ids = [
                getattr(mem, "id", None) for mem in memories if getattr(mem, "id", None)
            ]
            retrieved_tiers = {
                getattr(mem, "id", None): getattr(mem, "tier", None)
                for mem in memories
                if getattr(mem, "id", None) and getattr(mem, "tier", None)
            }

            # Format memories with confidence markers based on consolidation
            context_parts = ["[Previous learnings relevant to this debate:]"]
            for mem in memories[:3]:  # Top 3 most relevant
                content = mem.content[:200] if hasattr(mem, "content") else str(mem)[:200]
                tier = mem.tier.value if hasattr(mem, "tier") else "unknown"
                # Consolidation score indicates reliability
                consolidation = getattr(mem, "consolidation_score", 0.5)
                confidence = (
                    "high" if consolidation > 0.7 else "medium" if consolidation > 0.4 else "low"
                )
                context_parts.append(f"- [{tier}|{confidence}] {content}")

            context = "\n".join(context_parts)
            self._continuum_context_cache = context
            logger.info(
                f"  [continuum] Retrieved {len(memories)} relevant memories for domain '{domain}'"
            )
            return context, retrieved_ids, retrieved_tiers
        except (AttributeError, TypeError, ValueError) as e:
            # Expected errors from memory system
            logger.warning(f"  [continuum] Memory retrieval error: {e}")
            return "", [], {}
        except (KeyError, IndexError, RuntimeError, OSError) as e:
            # Unexpected error - log with more detail but don't crash debate
            logger.warning(f"  [continuum] Unexpected memory error (type={type(e).__name__}): {e}")
            return "", [], {}

    async def refresh_evidence_for_round(
        self,
        combined_text: str,
        evidence_collector: Any,
        task: str,
        evidence_store_callback: Optional[Callable[..., Any]] = None,
    ) -> tuple[int, Any]:
        """Refresh evidence based on claims made during a debate round.

        Called after the critique phase to gather fresh evidence for claims
        that emerged in proposals and critiques.

        Args:
            combined_text: Combined text from proposals and critiques
            evidence_collector: EvidenceCollector instance
            task: The debate task
            evidence_store_callback: Optional callback to store evidence in memory

        Returns:
            Tuple of:
            - Number of new evidence snippets added
            - Updated evidence pack (or None)
        """
        if not evidence_collector:
            return 0, None

        try:
            # Extract claims from the combined text
            claims = evidence_collector.extract_claims_from_text(combined_text)
            if not claims:
                return 0, None

            logger.debug(f"evidence_refresh extracting from {len(claims)} claims")

            # Collect evidence for the claims
            evidence_pack = await evidence_collector.collect_for_claims(claims)

            if not evidence_pack.snippets:
                return 0, None

            # Merge with existing evidence pack, avoiding duplicates
            if self._research_evidence_pack:
                existing_ids = {s.id for s in self._research_evidence_pack.snippets}
                new_snippets = [s for s in evidence_pack.snippets if s.id not in existing_ids]
                self._research_evidence_pack.snippets.extend(new_snippets)
                self._research_evidence_pack.total_searched += evidence_pack.total_searched
            else:
                self._research_evidence_pack = evidence_pack

            # Store in memory for future debates
            if (
                evidence_pack.snippets
                and evidence_store_callback
                and callable(evidence_store_callback)
            ):
                evidence_store_callback(evidence_pack.snippets, task)

            return len(evidence_pack.snippets), self._research_evidence_pack

        except Exception as e:
            logger.warning(f"Evidence refresh failed: {e}")
            return 0, None

"""
Context gathering orchestrator for debate research and evidence collection.

Main orchestration class that coordinates source fetching, content processing,
caching, and ranking to provide comprehensive context for debates.

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
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Callable

from .cache import (
    ContextCache,
)
from .processors import ContentProcessor
from .sources import SourceFetcher, EVIDENCE_TIMEOUT, TRENDING_TIMEOUT

if TYPE_CHECKING:
    from aragora.rlm.compressor import HierarchicalCompressor

logger = logging.getLogger(__name__)

# Configurable timeouts (in seconds)
CONTEXT_GATHER_TIMEOUT = float(os.getenv("ARAGORA_CONTEXT_TIMEOUT", "150.0"))

# Check for Knowledge Mound availability
_KnowledgeMound: type | None = None

try:
    from aragora.knowledge.mound import KnowledgeMound as _ImportedKnowledgeMound

    HAS_KNOWLEDGE_MOUND = True
    _KnowledgeMound = _ImportedKnowledgeMound
except ImportError:
    HAS_KNOWLEDGE_MOUND = False

# Alias for cleaner usage
KnowledgeMound = _KnowledgeMound

# Check for Threat Intelligence Enrichment availability
_ThreatIntelEnrichment: type | None = None

try:
    from aragora.security.threat_intel_enrichment import (
        ThreatIntelEnrichment as _ImportedThreatIntelEnrichment,
        ENRICHMENT_ENABLED as THREAT_INTEL_ENABLED,
    )

    HAS_THREAT_INTEL = True
    _ThreatIntelEnrichment = _ImportedThreatIntelEnrichment
except ImportError:
    HAS_THREAT_INTEL = False
    THREAT_INTEL_ENABLED = False

# Alias for cleaner usage
ThreatIntelEnrichment = _ThreatIntelEnrichment


class ContextGatherer:
    """
    Gathers context from multiple sources for debate grounding.

    Sources include:
    - Aragora project documentation (for self-referential debates)
    - Web search via EvidenceCollector
    - GitHub repositories
    - Local documentation
    - Pulse/trending topics from social platforms

    IMPORTANT: ContextGatherer should be instantiated ONCE PER DEBATE.
    It maintains internal caches keyed by task hash to prevent context leakage.
    Do not reuse a single ContextGatherer instance across multiple debates.

    Usage:
        # Create per-debate (done automatically by Arena.init_phases())
        gatherer = ContextGatherer(evidence_store_callback=store_evidence)
        context = await gatherer.gather_all(task="Discuss AI safety")

        # Clear cache if reusing (not recommended)
        gatherer.clear_cache()
    """

    def __init__(
        self,
        evidence_store_callback: Callable[..., Any] | None = None,
        prompt_builder: Any | None = None,
        project_root: Path | None = None,
        enable_rlm_compression: bool = True,
        rlm_compressor: Optional["HierarchicalCompressor"] = None,
        rlm_compression_threshold: int = 3000,  # Chars above which to use RLM
        enable_knowledge_grounding: bool = True,
        knowledge_mound: Any | None = None,
        knowledge_workspace_id: str | None = None,
        enable_belief_guidance: bool = True,
        enable_threat_intel_enrichment: bool = True,
        threat_intel_enrichment: Any | None = None,
        enable_trending_context: bool = True,
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
            enable_knowledge_grounding: Whether to auto-query Knowledge Mound for context.
            knowledge_mound: Optional pre-configured KnowledgeMound instance.
            knowledge_workspace_id: Workspace ID for knowledge queries (default: 'debate').
            enable_belief_guidance: Whether to inject historical cruxes from similar debates.
            enable_threat_intel_enrichment: Whether to enrich security topics with threat intel.
            threat_intel_enrichment: Optional pre-configured ThreatIntelEnrichment instance.
            enable_trending_context: Whether to gather Pulse trending context.
        """
        self._evidence_store_callback = evidence_store_callback
        self._prompt_builder = prompt_builder
        self._project_root = project_root or Path(__file__).parent.parent.parent.parent

        # Initialize cache
        self._cache = ContextCache()

        # Check for trending context disable
        disable_trending = os.getenv("ARAGORA_DISABLE_TRENDING", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._enable_trending_context = enable_trending_context and not disable_trending
        if disable_trending:
            logger.info(
                "[pulse] ContextGatherer: Trending context disabled via ARAGORA_DISABLE_TRENDING"
            )

        # Knowledge Mound configuration for auto-grounding
        self._enable_knowledge_grounding = enable_knowledge_grounding and HAS_KNOWLEDGE_MOUND
        self._knowledge_mound = knowledge_mound
        self._knowledge_workspace_id = knowledge_workspace_id or "debate"

        if self._enable_knowledge_grounding and KnowledgeMound is not None:
            if not self._knowledge_mound:
                try:
                    from aragora.knowledge.mound import get_knowledge_mound

                    self._knowledge_mound = get_knowledge_mound(
                        workspace_id=self._knowledge_workspace_id,
                        auto_initialize=True,
                    )
                    logger.info(
                        "[knowledge] ContextGatherer: Knowledge Mound enabled (workspace=%s)",
                        self._knowledge_workspace_id,
                    )
                except (RuntimeError, ValueError, OSError) as e:
                    logger.warning("[knowledge] Failed to initialize Knowledge Mound: %s", e)
                    self._enable_knowledge_grounding = False
                except ImportError:
                    # Fallback: instantiate directly if singleton helper unavailable
                    try:
                        if KnowledgeMound is None:
                            raise RuntimeError(
                                "KnowledgeMound not available - knowledge module not loaded"
                            )
                        self._knowledge_mound = KnowledgeMound(
                            workspace_id=self._knowledge_workspace_id
                        )
                        logger.info(
                            "[knowledge] ContextGatherer: Knowledge Mound enabled (workspace=%s)",
                            self._knowledge_workspace_id,
                        )
                    except (RuntimeError, ValueError, OSError) as e:
                        logger.warning("[knowledge] Failed to initialize Knowledge Mound: %s", e)
                        self._enable_knowledge_grounding = False
                except (TypeError, AttributeError, KeyError, ImportError, ConnectionError) as e:
                    logger.warning(
                        "[knowledge] Unexpected error initializing Knowledge Mound: %s", e
                    )
                    self._enable_knowledge_grounding = False
            else:
                logger.info("[knowledge] ContextGatherer: Using provided Knowledge Mound instance")

        # Belief guidance configuration for crux injection
        self._enable_belief_guidance = enable_belief_guidance
        self._belief_analyzer = None
        if self._enable_belief_guidance:
            try:
                from aragora.debate.phases.belief_analysis import DebateBeliefAnalyzer

                self._belief_analyzer = DebateBeliefAnalyzer()
                logger.info("[belief] ContextGatherer: Belief guidance enabled for crux injection")
            except ImportError:
                logger.debug("[belief] Belief analyzer module not available")
                self._enable_belief_guidance = False
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
                logger.warning("[belief] Failed to initialize belief analyzer: %s", e)
                self._enable_belief_guidance = False

        # Threat intelligence enrichment for security topics
        self._enable_threat_intel = (
            enable_threat_intel_enrichment and HAS_THREAT_INTEL and THREAT_INTEL_ENABLED
        )
        self._threat_intel_enrichment = threat_intel_enrichment
        if self._enable_threat_intel and ThreatIntelEnrichment is not None:
            if not self._threat_intel_enrichment:
                try:
                    self._threat_intel_enrichment = ThreatIntelEnrichment()
                    logger.info(
                        "[threat_intel] ContextGatherer: Threat intel enrichment enabled "
                        "for security topics"
                    )
                except (RuntimeError, ValueError, OSError) as e:
                    logger.warning("[threat_intel] Failed to initialize enrichment: %s", e)
                    self._enable_threat_intel = False
                except (TypeError, AttributeError, KeyError, ImportError, ConnectionError) as e:
                    logger.warning("[threat_intel] Unexpected error initializing enrichment: %s", e)
                    self._enable_threat_intel = False
            else:
                logger.info("[threat_intel] ContextGatherer: Using provided enrichment instance")

        # Initialize content processor
        self._processor = ContentProcessor(
            project_root=self._project_root,
            enable_rlm_compression=enable_rlm_compression,
            rlm_compressor=rlm_compressor,
            rlm_compression_threshold=rlm_compression_threshold,
            knowledge_mound=self._knowledge_mound,
        )

        # Initialize source fetcher
        self._source_fetcher = SourceFetcher(
            project_root=self._project_root,
            knowledge_mound=self._knowledge_mound if self._enable_knowledge_grounding else None,
            knowledge_workspace_id=self._knowledge_workspace_id,
            threat_intel_enrichment=(
                self._threat_intel_enrichment if self._enable_threat_intel else None
            ),
            belief_analyzer=self._belief_analyzer if self._enable_belief_guidance else None,
            enable_trending_context=self._enable_trending_context,
        )

    @property
    def evidence_pack(self) -> Any | None:
        """Get the most recent cached evidence pack.

        For task-specific evidence, use get_evidence_pack(task) instead.
        """
        return self._cache.get_latest_evidence_pack()

    def get_evidence_pack(self, task: str) -> Any | None:
        """Get the cached evidence pack for a specific task."""
        return self._cache.get_evidence_pack(task)

    def set_prompt_builder(self, prompt_builder: Any) -> None:
        """Set or update the prompt builder reference."""
        self._prompt_builder = prompt_builder

    def _get_task_hash(self, task: str) -> str:
        """Generate a cache key from task to prevent cache leaks between debates."""
        return ContextCache.get_task_hash(task)

    async def gather_all(self, task: str, timeout: float | None = None) -> str:
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
        # Check cache WITH task identity to prevent leaks between debates
        cached = self._cache.get_context(task)
        if cached:
            return cached

        timeout = timeout or CONTEXT_GATHER_TIMEOUT
        context_parts = []

        async def _gather_with_timeout():
            nonlocal context_parts

            # 1. Primary: Claude's web search (best quality research)
            claude_ctx = await self._source_fetcher.gather_claude_web_search(task)
            if claude_ctx:
                context_parts.append(claude_ctx)

            # 2. Gather Aragora context (local files, fast)
            aragora_ctx = await self.gather_aragora_context(task)
            if aragora_ctx:
                context_parts.append(aragora_ctx)

            # 3. Gather trending context for real-time relevance (if enabled)
            trending_task = None
            if self._enable_trending_context:
                trending_task = asyncio.create_task(self._gather_trending_with_timeout())

            # 4. Gather knowledge mound context for institutional knowledge
            knowledge_task = asyncio.create_task(
                self._source_fetcher.gather_knowledge_mound_with_timeout(task)
            )

            # 5. Gather belief crux context for debate guidance (fast, cached)
            belief_task = asyncio.create_task(self._source_fetcher.gather_belief_with_timeout(task))

            # 6. Gather culture patterns for organizational learning
            culture_task = asyncio.create_task(
                self._source_fetcher.gather_culture_with_timeout(task)
            )

            # 7. Gather threat intelligence context for security topics
            threat_intel_task = asyncio.create_task(
                self._source_fetcher.gather_threat_intel_with_timeout(task)
            )

            # 8. Gather additional evidence in parallel (fallback if Claude search weak)
            tasks = [knowledge_task, belief_task, culture_task, threat_intel_task]
            if trending_task is not None:
                tasks.append(trending_task)

            if not claude_ctx or len(claude_ctx) < 500:
                evidence_task = asyncio.create_task(self._gather_evidence_with_timeout(task))
                tasks.insert(0, evidence_task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, str) and result:
                    context_parts.append(result)
                elif isinstance(result, asyncio.TimeoutError):
                    logger.warning("Context gathering subtask timed out")
                elif isinstance(result, Exception):
                    logger.debug("Context gathering subtask failed: %s", result)

        try:
            await asyncio.wait_for(_gather_with_timeout(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Context gathering timed out after %ss, using partial results", timeout)

        if context_parts:
            result = "\n\n".join(context_parts)
            self._cache.set_context(task, result)
            return result
        else:
            return "No research context available."

    async def _gather_evidence_with_timeout(self, task: str) -> str | None:
        """Gather evidence with timeout protection."""
        try:
            ctx, evidence_pack = await asyncio.wait_for(
                self._source_fetcher.gather_evidence_context(
                    task,
                    evidence_store_callback=self._evidence_store_callback,
                    prompt_builder=self._prompt_builder,
                ),
                timeout=EVIDENCE_TIMEOUT,
            )
            if evidence_pack:
                self._cache.set_evidence_pack(task, evidence_pack)
            return ctx
        except asyncio.TimeoutError:
            logger.warning("Evidence collection timed out after %ss", EVIDENCE_TIMEOUT)
            return None

    async def _gather_trending_with_timeout(self) -> str | None:
        """Gather trending context with timeout protection."""
        try:
            ctx, topics = await asyncio.wait_for(
                self._source_fetcher.gather_trending_context(
                    prompt_builder=self._prompt_builder,
                ),
                timeout=TRENDING_TIMEOUT,
            )
            if topics:
                self._cache.set_trending_topics(topics)
            return ctx
        except asyncio.TimeoutError:
            logger.warning("Trending context timed out after %ss", TRENDING_TIMEOUT)
            return None

    async def gather_aragora_context(self, task: str) -> str | None:
        """
        Gather Aragora-specific documentation context if task is relevant.

        Only activates for tasks mentioning Aragora, multi-agent debates,
        decision stress-tests, nomic loop, or the debate framework.

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted documentation context, or None if not relevant.
        """
        return await self._processor.gather_aragora_context(task)

    async def gather_evidence_context(self, task: str) -> str | None:
        """
        Gather evidence from web, GitHub, and local docs connectors.

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted evidence context, or None if unavailable.
        """
        ctx, evidence_pack = await self._source_fetcher.gather_evidence_context(
            task,
            evidence_store_callback=self._evidence_store_callback,
            prompt_builder=self._prompt_builder,
        )
        if evidence_pack:
            self._cache.set_evidence_pack(task, evidence_pack)
        return ctx

    async def gather_trending_context(self) -> str | None:
        """
        Gather pulse/trending context from social platforms.

        Returns:
            Formatted trending topics context, or None if unavailable.
        """
        ctx, topics = await self._source_fetcher.gather_trending_context(
            prompt_builder=self._prompt_builder,
        )
        if topics:
            self._cache.set_trending_topics(topics)
        return ctx

    def get_trending_topics(self) -> list[Any]:
        """Get cached trending topics.

        Returns:
            List of TrendingTopic objects from the last gather_trending_context call.
        """
        return self._cache.get_trending_topics()

    async def gather_knowledge_mound_context(self, task: str) -> str | None:
        """
        Query Knowledge Mound for relevant facts and evidence.

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted knowledge context, or None if unavailable.
        """
        return await self._source_fetcher.gather_knowledge_mound_context(task)

    async def gather_threat_intel_context(self, task: str) -> str | None:
        """
        Gather threat intelligence context for security-related topics.

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted threat intelligence context, or None if not applicable.
        """
        return await self._source_fetcher.gather_threat_intel_context(task)

    async def gather_belief_crux_context(
        self,
        task: str,
        messages: list | None = None,
        top_k_cruxes: int = 3,
    ) -> str | None:
        """Gather crux claims from belief network analysis.

        Args:
            task: The debate task description.
            messages: Optional list of debate messages to analyze.
            top_k_cruxes: Number of top crux claims to extract.

        Returns:
            Formatted crux context string, or None if not available.
        """
        return await self._source_fetcher.gather_belief_crux_context(task, messages, top_k_cruxes)

    async def gather_culture_patterns_context(
        self,
        task: str,
        workspace_id: str | None = None,
    ) -> str | None:
        """
        Gather learned culture patterns from Knowledge Mound.

        Args:
            task: The debate task description.
            workspace_id: Optional workspace filter.

        Returns:
            Formatted culture patterns context, or None if unavailable.
        """
        return await self._source_fetcher.gather_culture_patterns_context(task, workspace_id)

    def clear_cache(self, task: str | None = None) -> None:
        """Clear cached context, optionally for a specific task.

        Args:
            task: If provided, only clear cache for this specific task.
                  If None, clear all cached context.
        """
        self._cache.clear(task)

    def get_continuum_context(
        self,
        continuum_memory: Any,
        domain: str,
        task: str,
        include_glacial_insights: bool = True,
        tenant_id: str | None = None,
        auth_context: Any | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Retrieve relevant memories from ContinuumMemory for debate context.

        Args:
            continuum_memory: ContinuumMemory instance to query
            domain: The debate domain (e.g., "programming", "ethics")
            task: The debate task description
            include_glacial_insights: Whether to include long-term glacial tier insights

        Returns:
            Tuple of:
            - Formatted context string
            - List of retrieved memory IDs (for outcome tracking)
            - Dict mapping memory ID to tier (for analytics)
        """
        # Check cache first
        cached = self._cache.get_continuum_context(task)
        if cached:
            return cached, [], {}

        context, ids, tiers = self._processor.get_continuum_context(
            continuum_memory,
            domain,
            task,
            include_glacial_insights,
            tenant_id=tenant_id,
            auth_context=auth_context,
        )

        if context:
            self._cache.set_continuum_context(task, context)

        return context, ids, tiers

    async def refresh_evidence_for_round(
        self,
        combined_text: str,
        evidence_collector: Any,
        task: str,
        evidence_store_callback: Callable[..., Any] | None = None,
    ) -> tuple[int, Any]:
        """Refresh evidence based on claims made during a debate round.

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
        callback = evidence_store_callback or self._evidence_store_callback
        count, evidence_pack = await self._processor.refresh_evidence_for_round(
            combined_text, evidence_collector, task, callback
        )

        if evidence_pack:
            merged = self._cache.merge_evidence_pack(task, evidence_pack)
            return count, merged

        return count, evidence_pack

    async def query_knowledge_with_true_rlm(
        self,
        task: str,
        max_items: int = 10,
    ) -> str | None:
        """
        Query Knowledge Mound using TRUE RLM for better answer quality.

        Args:
            task: The debate task/query
            max_items: Maximum knowledge items to include in context

        Returns:
            Synthesized answer from knowledge, or None if unavailable
        """
        if not self._enable_knowledge_grounding or not self._knowledge_mound:
            return None

        # Try TRUE RLM first
        result = await self._processor.query_knowledge_with_true_rlm(
            task,
            self._knowledge_mound,
            self._knowledge_workspace_id,
            max_items,
        )

        if result:
            return result

        # Fall back to standard knowledge query
        return await self.gather_knowledge_mound_context(task)

    # Expose internal components for advanced usage
    @property
    def cache(self) -> ContextCache:
        """Get the internal cache for advanced operations."""
        return self._cache

    @property
    def processor(self) -> ContentProcessor:
        """Get the content processor for advanced operations."""
        return self._processor

    @property
    def source_fetcher(self) -> SourceFetcher:
        """Get the source fetcher for advanced operations."""
        return self._source_fetcher


# Re-export for backwards compatibility
__all__ = [
    "ContextGatherer",
    "CONTEXT_GATHER_TIMEOUT",
    "HAS_KNOWLEDGE_MOUND",
    "HAS_THREAT_INTEL",
    "THREAT_INTEL_ENABLED",
]

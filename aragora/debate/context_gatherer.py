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
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Type

if TYPE_CHECKING:
    from aragora.rlm.compressor import HierarchicalCompressor

# Check for RLM availability (use factory for consistent initialization)
try:
    from aragora.rlm import get_rlm, get_compressor, HAS_OFFICIAL_RLM

    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HAS_OFFICIAL_RLM = False
    get_rlm: Optional[Callable[..., Any]] = None
    get_compressor: Optional[Callable[..., Any]] = None

# Check for Knowledge Mound availability
try:
    from aragora.knowledge.mound import KnowledgeMound

    HAS_KNOWLEDGE_MOUND = True
except ImportError:
    HAS_KNOWLEDGE_MOUND = False
    KnowledgeMound: Optional[Type[Any]] = None

logger = logging.getLogger(__name__)

# Configurable timeouts (in seconds)
# Increased timeouts to allow Claude web search to complete
CONTEXT_GATHER_TIMEOUT = float(os.getenv("ARAGORA_CONTEXT_TIMEOUT", "150.0"))
CLAUDE_SEARCH_TIMEOUT = float(os.getenv("ARAGORA_CLAUDE_SEARCH_TIMEOUT", "120.0"))
EVIDENCE_TIMEOUT = float(os.getenv("ARAGORA_EVIDENCE_TIMEOUT", "30.0"))
TRENDING_TIMEOUT = float(os.getenv("ARAGORA_TRENDING_TIMEOUT", "5.0"))
KNOWLEDGE_MOUND_TIMEOUT = float(os.getenv("ARAGORA_KNOWLEDGE_MOUND_TIMEOUT", "10.0"))
BELIEF_CRUX_TIMEOUT = float(os.getenv("ARAGORA_BELIEF_CRUX_TIMEOUT", "5.0"))
THREAT_INTEL_TIMEOUT = float(os.getenv("ARAGORA_THREAT_INTEL_TIMEOUT", "10.0"))

# Check for Threat Intelligence Enrichment availability
try:
    from aragora.security.threat_intel_enrichment import (
        ThreatIntelEnrichment,
        ENRICHMENT_ENABLED as THREAT_INTEL_ENABLED,
    )

    HAS_THREAT_INTEL = True
except ImportError:
    HAS_THREAT_INTEL = False
    THREAT_INTEL_ENABLED = False
    ThreatIntelEnrichment: Optional[Type[Any]] = None


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
        evidence_store_callback: Optional[Callable[..., Any]] = None,
        prompt_builder: Optional[Any] = None,
        project_root: Optional[Path] = None,
        enable_rlm_compression: bool = True,
        rlm_compressor: Optional["HierarchicalCompressor"] = None,
        rlm_compression_threshold: int = 3000,  # Chars above which to use RLM
        enable_knowledge_grounding: bool = True,
        knowledge_mound: Optional[Any] = None,
        knowledge_workspace_id: Optional[str] = None,
        enable_belief_guidance: bool = True,
        enable_threat_intel_enrichment: bool = True,
        threat_intel_enrichment: Optional[Any] = None,
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
        """
        self._evidence_store_callback = evidence_store_callback
        self._prompt_builder = prompt_builder
        self._project_root = project_root or Path(__file__).parent.parent.parent

        # Cache for evidence pack (keyed by task hash to prevent leaks between debates)
        self._research_evidence_pack: dict[str, Any] = {}

        # Cache for research context (keyed by task hash to prevent leaks between debates)
        self._research_context_cache: dict[str, str] = {}

        # Cache for continuum memory context (keyed by task hash to prevent leaks between debates)
        self._continuum_context_cache: dict[str, str] = {}

        # Cache for trending topics (TrendingTopic objects, not just formatted string)
        self._trending_topics_cache: list[Any] = []

        # RLM configuration - use factory for consistent initialization
        self._enable_rlm = enable_rlm_compression and HAS_RLM
        self._rlm_compressor = rlm_compressor
        self._aragora_rlm: Optional[Any] = None
        self._rlm_threshold = rlm_compression_threshold

        if self._enable_rlm and get_rlm is not None:
            # Use factory to get AragoraRLM (routes to TRUE RLM when available)
            try:
                self._aragora_rlm = get_rlm()
                if HAS_OFFICIAL_RLM:
                    logger.info(
                        "[rlm] ContextGatherer: TRUE RLM enabled via factory "
                        "(REPL-based, model writes code to examine context)"
                    )
                else:
                    logger.info(
                        "[rlm] ContextGatherer: AragoraRLM enabled via factory "
                        "(will use compression fallback since official RLM not installed)"
                    )
            except ImportError as e:
                # Expected: RLM module not installed
                logger.debug(f"[rlm] RLM module not available: {e}")
            except (RuntimeError, ValueError) as e:
                # Expected: RLM initialization issues
                logger.warning(f"[rlm] Failed to initialize RLM: {e}")
            except Exception as e:
                # Unexpected error
                logger.warning(f"[rlm] Unexpected error getting RLM from factory: {e}")

            # Fallback: get compressor from factory (compression-only)
            if not self._rlm_compressor and get_compressor is not None:
                try:
                    self._rlm_compressor = get_compressor()
                    logger.debug(
                        "[rlm] ContextGatherer: HierarchicalCompressor fallback via factory"
                    )
                except ImportError as e:
                    # Expected: compressor module not available
                    logger.debug(f"[rlm] Compressor module not available: {e}")
                except (RuntimeError, ValueError) as e:
                    # Expected: compressor initialization issues
                    logger.warning(f"[rlm] Failed to initialize compressor: {e}")
                except Exception as e:
                    # Unexpected error
                    logger.warning(f"[rlm] Unexpected error getting compressor: {e}")

        # Knowledge Mound configuration for auto-grounding
        self._enable_knowledge_grounding = enable_knowledge_grounding and HAS_KNOWLEDGE_MOUND
        self._knowledge_mound = knowledge_mound
        self._knowledge_workspace_id = knowledge_workspace_id or "debate"

        if self._enable_knowledge_grounding and KnowledgeMound is not None:
            if not self._knowledge_mound:
                try:
                    self._knowledge_mound = KnowledgeMound(
                        workspace_id=self._knowledge_workspace_id
                    )
                    logger.info(
                        f"[knowledge] ContextGatherer: Knowledge Mound enabled "
                        f"(workspace={self._knowledge_workspace_id})"
                    )
                except (RuntimeError, ValueError, OSError) as e:
                    # Expected: knowledge mound initialization issues
                    logger.warning(f"[knowledge] Failed to initialize Knowledge Mound: {e}")
                    self._enable_knowledge_grounding = False
                except Exception as e:
                    # Unexpected error
                    logger.warning(
                        f"[knowledge] Unexpected error initializing Knowledge Mound: {e}"
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
            except Exception as e:
                logger.warning(f"[belief] Failed to initialize belief analyzer: {e}")
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
                    logger.warning(f"[threat_intel] Failed to initialize enrichment: {e}")
                    self._enable_threat_intel = False
                except Exception as e:
                    logger.warning(f"[threat_intel] Unexpected error initializing enrichment: {e}")
                    self._enable_threat_intel = False
            else:
                logger.info("[threat_intel] ContextGatherer: Using provided enrichment instance")

    @property
    def evidence_pack(self) -> Optional[Any]:
        """Get the most recent cached evidence pack.

        For task-specific evidence, use get_evidence_pack(task) instead.
        """
        if not self._research_evidence_pack:
            return None
        # Return the most recently added pack for backward compatibility
        # In practice, callers should use get_evidence_pack(task) for isolation
        if self._research_evidence_pack:
            # Return last added pack (dict preserves insertion order in Python 3.7+)
            return list(self._research_evidence_pack.values())[-1]
        return None

    def get_evidence_pack(self, task: str) -> Optional[Any]:
        """Get the cached evidence pack for a specific task."""
        task_hash = self._get_task_hash(task)
        return self._research_evidence_pack.get(task_hash)

    def set_prompt_builder(self, prompt_builder: Any) -> None:
        """Set or update the prompt builder reference."""
        self._prompt_builder = prompt_builder

    def _get_task_hash(self, task: str) -> str:
        """Generate a cache key from task to prevent cache leaks between debates."""
        return hashlib.sha256(task.encode()).hexdigest()[:16]

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
        # Check cache WITH task identity to prevent leaks between debates
        task_hash = self._get_task_hash(task)
        if task_hash in self._research_context_cache:
            return self._research_context_cache[task_hash]

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

            # 4. Gather knowledge mound context for institutional knowledge
            knowledge_task = asyncio.create_task(self._gather_knowledge_mound_with_timeout(task))

            # 5. Gather belief crux context for debate guidance (fast, cached)
            belief_task = asyncio.create_task(self._gather_belief_with_timeout(task))

            # 6. Gather culture patterns for organizational learning
            culture_task = asyncio.create_task(self._gather_culture_with_timeout(task))

            # 7. Gather threat intelligence context for security topics
            threat_intel_task = asyncio.create_task(self._gather_threat_intel_with_timeout(task))

            # 8. Gather additional evidence in parallel (fallback if Claude search weak)
            if not claude_ctx or len(claude_ctx) < 500:
                evidence_task = asyncio.create_task(self._gather_evidence_with_timeout(task))
                results = await asyncio.gather(
                    evidence_task,
                    trending_task,
                    knowledge_task,
                    belief_task,
                    culture_task,
                    threat_intel_task,
                    return_exceptions=True,
                )
            else:
                # Still wait for trending, knowledge, belief, culture, and threat intel
                results = await asyncio.gather(
                    trending_task,
                    knowledge_task,
                    belief_task,
                    culture_task,
                    threat_intel_task,
                    return_exceptions=True,
                )

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
            result = "\n\n".join(context_parts)
            self._research_context_cache[task_hash] = result
            return result
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
        except (ConnectionError, OSError) as e:
            # Expected: network or API issues
            logger.warning(f"[research] Claude web search network error: {e}")
            return None
        except (ValueError, RuntimeError) as e:
            # Expected: API or response processing issues
            logger.warning(f"[research] Claude web search failed: {e}")
            return None
        except Exception as e:
            # Unexpected error
            logger.warning(f"[research] Unexpected error in Claude web search: {e}")
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

    async def _gather_knowledge_mound_with_timeout(self, task: str) -> Optional[str]:
        """Gather knowledge mound context with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.gather_knowledge_mound_context(task), timeout=KNOWLEDGE_MOUND_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Knowledge mound context timed out after {KNOWLEDGE_MOUND_TIMEOUT}s")
            return None

    async def _gather_threat_intel_with_timeout(self, task: str) -> Optional[str]:
        """Gather threat intelligence context with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.gather_threat_intel_context(task), timeout=THREAT_INTEL_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Threat intel context timed out after {THREAT_INTEL_TIMEOUT}s")
            return None

    async def gather_threat_intel_context(self, task: str) -> Optional[str]:
        """
        Gather threat intelligence context for security-related topics.

        When the debate topic involves security (vulnerabilities, CVEs, attacks, etc.),
        this method enriches the context with relevant threat intelligence:
        - CVE vulnerability details
        - IP/domain/URL reputation
        - File hash malware analysis
        - Attack patterns and mitigations

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted threat intelligence context, or None if not applicable.
        """
        if not self._enable_threat_intel or not self._threat_intel_enrichment:
            return None

        try:
            # Check if the topic is security-related first (fast, local check)
            if not self._threat_intel_enrichment.is_security_topic(task):
                logger.debug("[threat_intel] Task is not security-related, skipping enrichment")
                return None

            # Enrich with threat intelligence
            context = await self._threat_intel_enrichment.enrich_context(
                topic=task,
                existing_context="",  # Could include additional context if available
            )

            if context:
                formatted = self._threat_intel_enrichment.format_for_debate(context)
                indicator_count = len(context.indicators)
                cve_count = len(context.relevant_cves)
                logger.info(
                    f"[threat_intel] Enriched security context with "
                    f"{indicator_count} indicators and {cve_count} CVEs"
                )
                return formatted

            logger.debug("[threat_intel] No enrichment data available for topic")
            return None

        except (ConnectionError, OSError) as e:
            # Expected: network or API issues
            logger.warning(f"[threat_intel] Enrichment network error: {e}")
            return None
        except (ValueError, KeyError, AttributeError) as e:
            # Expected: data format or access issues
            logger.warning(f"[threat_intel] Enrichment failed: {e}")
            return None
        except Exception as e:
            # Unexpected error
            logger.warning(f"[threat_intel] Unexpected error in enrichment: {e}")
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

        except (OSError, IOError) as e:
            # Expected: file system issues reading docs
            logger.warning(f"Failed to load Aragora context (file error): {e}")
        except (ValueError, RuntimeError) as e:
            # Expected: compression or parsing issues
            logger.warning(f"Failed to load Aragora context: {e}")
        except Exception as e:
            # Unexpected error
            logger.warning(f"Unexpected error loading Aragora context: {e}")

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
                task_hash = self._get_task_hash(task)
                self._research_evidence_pack[task_hash] = evidence_pack

                # Update prompt builder if available
                if self._prompt_builder:
                    self._prompt_builder.set_evidence_pack(evidence_pack)

                # Store evidence via callback if provided
                if self._evidence_store_callback and callable(self._evidence_store_callback):
                    self._evidence_store_callback(evidence_pack.snippets, task)

                return f"## EVIDENCE CONTEXT\n{evidence_pack.to_context_string()}"

        except ImportError as e:
            # Expected: evidence collector module not available
            logger.debug(f"Evidence collector not available: {e}")
        except (ConnectionError, OSError) as e:
            # Expected: network or file system issues
            logger.warning(f"Evidence collection network/IO error: {e}")
        except (ValueError, RuntimeError) as e:
            # Expected: data processing issues
            logger.warning(f"Evidence collection failed: {e}")
        except Exception as e:
            # Unexpected error
            logger.warning(f"Unexpected error in evidence collection: {e}")

        return None

    async def gather_trending_context(self) -> Optional[str]:
        """
        Gather pulse/trending context from social platforms.

        Uses PulseManager with available ingestors:
        - Twitter
        - Hacker News
        - Reddit
        - GitHub Trending
        - Google Trends

        Also caches TrendingTopic objects for prompt_builder injection.

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
                # Cache TrendingTopic objects for PromptBuilder injection
                self._trending_topics_cache = list(topics)

                # Pass to prompt builder if available
                if self._prompt_builder:
                    self._prompt_builder.set_trending_topics(self._trending_topics_cache)
                    logger.debug(
                        f"[pulse] Injected {len(self._trending_topics_cache)} trending topics "
                        f"into PromptBuilder"
                    )

                trending_context = (
                    "## TRENDING CONTEXT\nCurrent trending topics that may be relevant:\n"
                )
                for t in topics[:5]:
                    trending_context += (
                        f"- {t.topic} ({t.platform}, {t.volume:,} engagement, {t.category})\n"
                    )
                return trending_context

        except ImportError as e:
            # Expected: pulse module not installed
            logger.debug(f"Pulse module not available: {e}")
        except (ConnectionError, OSError) as e:
            # Expected: network issues fetching trends
            logger.debug(f"Pulse context network error: {e}")
        except (ValueError, RuntimeError) as e:
            # Expected: API or data processing issues
            logger.debug(f"Pulse context unavailable: {e}")
        except Exception as e:
            # Unexpected error
            logger.warning(f"Unexpected error getting pulse context: {e}")

        return None

    def get_trending_topics(self) -> list[Any]:
        """Get cached trending topics.

        Returns:
            List of TrendingTopic objects from the last gather_trending_context call.
        """
        return self._trending_topics_cache

    async def gather_knowledge_mound_context(self, task: str) -> Optional[str]:
        """
        Query Knowledge Mound for relevant facts and evidence.

        Auto-grounds debates with institutional knowledge from:
        - Previous debate outcomes and consensus
        - Extracted facts from documents
        - Evidence snippets with provenance
        - Cross-debate patterns and insights

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted knowledge context, or None if unavailable.
        """
        if not self._enable_knowledge_grounding or not self._knowledge_mound:
            return None

        try:
            # Query knowledge mound for relevant items
            result = await self._knowledge_mound.query(
                query=task,
                sources=("all",),  # Query all knowledge sources
                limit=10,
            )

            if not result.items:
                logger.debug("[knowledge] No relevant knowledge found for task")
                return None

            # Format knowledge context
            context_parts = [
                "## KNOWLEDGE MOUND CONTEXT",
                "Relevant institutional knowledge from previous debates and analyses:",
                "",
            ]

            # Group by source type for better organization
            facts = []
            evidence = []
            insights = []

            for item in result.items:
                source = getattr(item, "source", None)
                source_name = (
                    source.value
                    if hasattr(source, "value")
                    else str(source)
                    if source
                    else "unknown"
                )
                content = item.content[:500] if item.content else ""
                confidence = getattr(item, "confidence", 0.5)

                if source_name in ("fact", "fact_store"):
                    facts.append((content, confidence))
                elif source_name in ("evidence", "evidence_store"):
                    evidence.append((content, confidence))
                else:
                    insights.append((content, confidence, source_name))

            # Format facts
            if facts:
                context_parts.append("### Verified Facts")
                for content, conf in facts[:3]:
                    conf_label = "HIGH" if conf > 0.7 else "MEDIUM" if conf > 0.4 else "LOW"
                    context_parts.append(f"- [{conf_label}] {content}")
                context_parts.append("")

            # Format evidence
            if evidence:
                context_parts.append("### Supporting Evidence")
                for content, conf in evidence[:3]:
                    context_parts.append(f"- {content}")
                context_parts.append("")

            # Format insights
            if insights:
                context_parts.append("### Related Insights")
                for content, conf, source in insights[:4]:
                    context_parts.append(f"- ({source}) {content}")
                context_parts.append("")

            if len(context_parts) <= 3:
                # Only header, no actual content
                return None

            logger.info(
                f"[knowledge] Injected {len(result.items)} knowledge items "
                f"({len(facts)} facts, {len(evidence)} evidence, {len(insights)} insights) "
                f"in {result.execution_time_ms:.0f}ms"
            )

            return "\n".join(context_parts)

        except (ConnectionError, OSError) as e:
            # Expected: storage or network issues
            logger.warning(f"[knowledge] Knowledge Mound query failed (IO error): {e}")
            return None
        except (ValueError, KeyError, AttributeError) as e:
            # Expected: data format or access issues
            logger.warning(f"[knowledge] Knowledge Mound query failed: {e}")
            return None
        except Exception as e:
            # Unexpected error
            logger.warning(f"[knowledge] Unexpected error in Knowledge Mound query: {e}")
            return None

    async def gather_belief_crux_context(
        self,
        task: str,
        messages: Optional[list] = None,
        top_k_cruxes: int = 3,
    ) -> Optional[str]:
        """Gather crux claims from belief network analysis.

        Analyzes debate messages (if provided) or queries historical debates
        for crux claims that can inform the current debate.

        Args:
            task: The debate task description.
            messages: Optional list of debate messages to analyze.
            top_k_cruxes: Number of top crux claims to extract.

        Returns:
            Formatted crux context string, or None if not available.
        """
        if not self._enable_belief_guidance or not self._belief_analyzer:
            return None

        try:
            # If messages provided, analyze them for cruxes
            if messages:
                result = self._belief_analyzer.analyze_messages(
                    messages,
                    top_k_cruxes=top_k_cruxes,
                )

                if result.analysis_error:
                    logger.debug(f"[belief] Crux analysis error: {result.analysis_error}")
                    return None

                if not result.cruxes:
                    return None

                context_parts = [
                    "## Key Crux Points (Belief Network Analysis)",
                    "",
                    "The following crux claims represent pivotal points of disagreement",
                    "that may determine the outcome of this debate:",
                    "",
                ]

                for i, crux in enumerate(result.cruxes[:top_k_cruxes], 1):
                    statement = crux.get("statement", crux.get("claim", ""))
                    confidence = crux.get("confidence", 0.5)
                    entropy = crux.get("entropy", 0.5)

                    conf_label = (
                        "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"
                    )
                    contested = " (CONTESTED)" if entropy > 0.8 else ""
                    context_parts.append(f"{i}. [{conf_label}{contested}] {statement}")

                if result.evidence_suggestions:
                    context_parts.extend(
                        [
                            "",
                            "### Evidence Needed",
                            "The following evidence would help resolve these cruxes:",
                        ]
                    )
                    for suggestion in result.evidence_suggestions[:3]:
                        context_parts.append(f"- {suggestion}")

                logger.info(
                    f"[belief] Extracted {len(result.cruxes)} cruxes from {len(messages)} messages"
                )

                return "\n".join(context_parts)

            # No messages - could query historical debates for similar topic cruxes
            # This would require a crux store, which could be added later
            return None

        except (ValueError, AttributeError) as e:
            # Expected: data format issues
            logger.debug(f"[belief] Crux gathering failed: {e}")
            return None
        except Exception as e:
            # Unexpected error
            logger.warning(f"[belief] Unexpected error gathering cruxes: {e}")
            return None

    async def _gather_belief_with_timeout(self, task: str) -> Optional[str]:
        """Gather belief crux context with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.gather_belief_crux_context(task),
                timeout=BELIEF_CRUX_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[belief] Crux gathering timed out after {BELIEF_CRUX_TIMEOUT}s")
            return None

    async def _gather_culture_with_timeout(self, task: str) -> Optional[str]:
        """Gather culture patterns context with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.gather_culture_patterns_context(task),
                timeout=5.0,  # Culture patterns should be fast (local query)
            )
        except asyncio.TimeoutError:
            logger.warning("[culture] Culture pattern gathering timed out")
            return None

    async def gather_culture_patterns_context(
        self,
        task: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Gather learned culture patterns from Knowledge Mound.

        Retrieves organizational patterns learned from previous debates:
        - Successful argumentation strategies
        - Consensus-building patterns
        - Domain-specific debate styles
        - Protocol effectiveness patterns

        Args:
            task: The debate task description.
            workspace_id: Optional workspace filter.

        Returns:
            Formatted culture patterns context, or None if unavailable.
        """
        try:
            # Use the mound's culture context API if available
            if self._knowledge_mound and hasattr(self._knowledge_mound, "get_culture_context"):
                ws_id = workspace_id or self._knowledge_workspace_id or "default"
                context = await self._knowledge_mound.get_culture_context(
                    org_id=ws_id,
                    task=task,
                    max_documents=5,
                )
                if context:
                    logger.info("[culture] Injected culture context from mound")
                    return context
                logger.debug("[culture] No relevant culture context from mound")
                return None

            # Fallback: use CultureAccumulator with mound reference
            if not self._knowledge_mound:
                logger.debug("[culture] Knowledge mound not available for culture patterns")
                return None

            from aragora.knowledge.mound.culture.accumulator import CultureAccumulator

            # Get or create accumulator from mound (or mound's internal accumulator)
            if (
                hasattr(self._knowledge_mound, "_culture_accumulator")
                and self._knowledge_mound._culture_accumulator
            ):
                accumulator = self._knowledge_mound._culture_accumulator
            else:
                accumulator = CultureAccumulator(mound=self._knowledge_mound)

            # Query for patterns relevant to this workspace
            ws_id = workspace_id or self._knowledge_workspace_id or "default"
            patterns = accumulator.get_patterns(
                workspace_id=ws_id,
                min_confidence=0.3,
            )

            if not patterns:
                logger.debug("[culture] No relevant culture patterns found")
                return None

            context_parts = [
                "## ORGANIZATIONAL CULTURE PATTERNS",
                "Learned patterns from previous debates in this workspace:",
                "",
            ]

            for pattern in patterns[:5]:  # Limit to top 5
                pattern_type = getattr(pattern, "pattern_type", None) or "general"
                description = getattr(pattern, "description", "") or ""
                confidence = getattr(pattern, "confidence", 0.5)
                observations = getattr(pattern, "observations", 0)

                if not description:
                    continue

                # Convert enum to string if needed
                if hasattr(pattern_type, "value"):
                    pattern_type = pattern_type.value

                conf_label = (
                    "Strong" if confidence > 0.7 else "Moderate" if confidence > 0.4 else "Emerging"
                )
                context_parts.append(
                    f"- **{str(pattern_type).title()}** [{conf_label}, {observations} uses]: {description}"
                )

            if len(context_parts) <= 3:
                return None

            logger.info(f"[culture] Injected {min(len(patterns), 5)} culture patterns")
            return "\n".join(context_parts)

        except ImportError:
            logger.debug("[culture] CultureAccumulator not available")
            return None
        except Exception as e:
            logger.warning(f"[culture] Failed to gather culture patterns: {e}")
            return None

    def clear_cache(self, task: Optional[str] = None) -> None:
        """Clear cached context, optionally for a specific task.

        Args:
            task: If provided, only clear cache for this specific task.
                  If None, clear all cached context.
        """
        if task is None:
            self._research_context_cache.clear()
            self._research_evidence_pack.clear()
            self._continuum_context_cache.clear()
            self._trending_topics_cache = []
        else:
            task_hash = self._get_task_hash(task)
            self._research_context_cache.pop(task_hash, None)
            self._research_evidence_pack.pop(task_hash, None)
            self._continuum_context_cache.pop(task_hash, None)

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
            return (
                content[: max_chars - 30] + "... [truncated]"
                if len(content) > max_chars
                else content
            )

        # PRIMARY: Try AragoraRLM (routes to TRUE RLM if available)
        if self._aragora_rlm:
            try:
                logger.debug(
                    "[rlm] Using AragoraRLM for compression (routes to TRUE RLM if available)"
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
                        f"({len(result.answer) / len(content) * 100:.0f}%) via {approach}"
                    )
                    return (
                        result.answer[:max_chars]
                        if len(result.answer) > max_chars
                        else result.answer
                    )

            except asyncio.TimeoutError:
                logger.debug("[rlm] AragoraRLM compression timed out")
            except (ValueError, RuntimeError) as e:
                # Expected: compression configuration or processing issues
                logger.debug(f"[rlm] AragoraRLM compression failed: {e}")
            except Exception as e:
                # Unexpected error
                logger.warning(f"[rlm] Unexpected error in AragoraRLM compression: {e}")

        # FALLBACK: Try direct HierarchicalCompressor (compression-only)
        if self._rlm_compressor:
            try:
                logger.debug(
                    "[rlm] Falling back to HierarchicalCompressor (compression-only, no TRUE RLM)"
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
                        f"({len(summary) / len(content) * 100:.0f}%) via HierarchicalCompressor"
                    )
                    return summary[:max_chars] if len(summary) > max_chars else summary

            except asyncio.TimeoutError:
                logger.debug("[rlm] HierarchicalCompressor timed out")
            except (ValueError, RuntimeError) as e:
                # Expected: compression configuration or processing issues
                logger.debug(f"[rlm] HierarchicalCompressor failed: {e}")
            except Exception as e:
                # Unexpected error
                logger.warning(f"[rlm] Unexpected error in HierarchicalCompressor: {e}")

        # FINAL FALLBACK: Simple truncation
        logger.debug("[rlm] All RLM approaches failed, using simple truncation")
        return (
            content[: max_chars - 30] + "... [truncated]" if len(content) > max_chars else content
        )

    async def _query_with_true_rlm(
        self,
        query: str,
        content: str,
        source_type: str = "documentation",
    ) -> Optional[str]:
        """
        Query content using TRUE RLM (REPL-based) when available.

        TRUE RLM allows the model to write code to examine context stored
        as Python variables in a REPL environment, rather than having the
        context stuffed into prompts.

        This is the PREFERRED method when the official `rlm` package is installed:
        - Model has agency in deciding how to query content
        - No information loss from truncation or compression
        - Model writes code like: `search_debate(context, r"consensus")`

        Falls back to compression-based query if TRUE RLM not available.

        Args:
            query: The question to answer about the content
            content: The content to query
            source_type: Type of content (for context hints)

        Returns:
            Answer from TRUE RLM, or None if not available
        """
        if not self._enable_rlm or not self._aragora_rlm:
            return None

        try:
            # Check if TRUE RLM is available (not just compression fallback)
            if HAS_RLM and HAS_OFFICIAL_RLM:
                logger.debug(
                    f"[rlm] Using TRUE RLM for query: '{query[:50]}...' "
                    f"on {len(content)} chars of {source_type}"
                )

                result = await asyncio.wait_for(
                    self._aragora_rlm.query(
                        query=query,
                        context=content,
                        strategy="auto",  # Let RLM decide: grep, partition, peek, etc.
                    ),
                    timeout=20.0,
                )

                if result.used_true_rlm and result.answer:
                    logger.debug(
                        f"[rlm] TRUE RLM query successful: {len(result.answer)} chars, "
                        f"confidence={result.confidence:.2f}"
                    )
                    return result.answer

            # TRUE RLM not available - fall back to compress_and_query
            logger.debug("[rlm] TRUE RLM not available for query, using compress_and_query")
            result = await asyncio.wait_for(
                self._aragora_rlm.compress_and_query(
                    query=query,
                    content=content,
                    source_type=source_type,
                ),
                timeout=15.0,
            )

            if result.answer:
                approach = "TRUE RLM" if result.used_true_rlm else "compression"
                logger.debug(f"[rlm] Query via {approach}: {len(result.answer)} chars")
                return result.answer

        except asyncio.TimeoutError:
            logger.debug(f"[rlm] TRUE RLM query timed out for: '{query[:30]}...'")
        except (ValueError, RuntimeError) as e:
            # Expected: query processing issues
            logger.debug(f"[rlm] TRUE RLM query failed: {e}")
        except Exception as e:
            # Unexpected error
            logger.warning(f"[rlm] Unexpected error in TRUE RLM query: {e}")

        return None

    async def query_knowledge_with_true_rlm(
        self,
        task: str,
        max_items: int = 10,
    ) -> Optional[str]:
        """
        Query Knowledge Mound using TRUE RLM for better answer quality.

        When TRUE RLM is available, creates a REPL environment where the
        model can write code to navigate and query knowledge items.

        Args:
            task: The debate task/query
            max_items: Maximum knowledge items to include in context

        Returns:
            Synthesized answer from knowledge, or None if unavailable
        """
        if not self._enable_knowledge_grounding or not self._knowledge_mound:
            return None

        if not (HAS_RLM and HAS_OFFICIAL_RLM):
            # TRUE RLM not available - use standard query
            return await self.gather_knowledge_mound_context(task)

        try:
            from aragora.rlm import get_repl_adapter

            adapter = get_repl_adapter()

            # Create REPL environment for knowledge queries
            env = adapter.create_repl_for_knowledge(
                mound=self._knowledge_mound,
                workspace_id=self._knowledge_workspace_id,
                content_id=f"km_{self._get_task_hash(task)}",
            )

            if not env:
                # TRUE RLM REPL failed - fall back to standard
                return await self.gather_knowledge_mound_context(task)

            # Get REPL prompt for agent
            repl_prompt = adapter.get_repl_prompt(
                content_id=f"km_{self._get_task_hash(task)}",
                content_type="knowledge",
            )

            logger.info(
                f"[rlm] Created TRUE RLM REPL environment for knowledge query: '{task[:50]}...'"
            )

            # Return the REPL prompt - the agent will use it to write code
            # that queries the knowledge programmatically
            return (
                "## KNOWLEDGE MOUND CONTEXT (TRUE RLM)\n"
                f"A REPL environment is available for knowledge queries.\n\n"
                f"{repl_prompt}\n"
            )

        except ImportError:
            logger.debug("[rlm] REPL adapter not available for knowledge queries")
        except Exception as e:
            logger.warning(f"[rlm] Failed to create knowledge REPL: {e}")

        # Fall back to standard knowledge query
        return await self.gather_knowledge_mound_context(task)

    def get_continuum_context(
        self,
        continuum_memory: Any,
        domain: str,
        task: str,
        include_glacial_insights: bool = True,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Retrieve relevant memories from ContinuumMemory for debate context.

        Uses the debate task and domain to query for related past learnings.
        Enhanced with tier-aware retrieval, confidence markers, and glacial insights.

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
        # Check task-keyed cache first
        task_hash = self._get_task_hash(task)
        if hasattr(self, "_continuum_context_cache") and task_hash in self._continuum_context_cache:
            return self._continuum_context_cache[task_hash], [], {}

        if not continuum_memory:
            return "", [], {}

        try:
            query = f"{domain}: {task[:200]}"
            all_memories = []
            retrieved_ids = []
            retrieved_tiers = {}

            # 1. Retrieve recent memories from fast/medium/slow tiers
            memories = continuum_memory.retrieve(
                query=query,
                limit=5,
                min_importance=0.3,
                include_glacial=False,  # Get recent memories first
            )
            all_memories.extend(memories)

            # 2. Also retrieve glacial tier insights for cross-session learning
            if include_glacial_insights and hasattr(continuum_memory, "get_glacial_insights"):
                glacial_insights = continuum_memory.get_glacial_insights(
                    topic=task[:100],
                    limit=3,
                    min_importance=0.4,  # Higher threshold for long-term patterns
                )
                if glacial_insights:
                    logger.info(
                        f"  [continuum] Retrieved {len(glacial_insights)} glacial insights "
                        f"for cross-session learning"
                    )
                    all_memories.extend(glacial_insights)

            if not all_memories:
                return "", [], {}

            # Track retrieved memory IDs and tiers for outcome updates and analytics
            retrieved_ids = [
                getattr(mem, "id", None) for mem in all_memories if getattr(mem, "id", None)
            ]
            retrieved_tiers = {
                getattr(mem, "id", None): getattr(mem, "tier", None)
                for mem in all_memories
                if getattr(mem, "id", None) and getattr(mem, "tier", None)
            }

            # Format memories with confidence markers based on consolidation
            context_parts = ["[Previous learnings relevant to this debate:]"]

            # Format recent memories (fast/medium/slow)
            recent_mems = [
                m
                for m in all_memories
                if getattr(m, "tier", None) and getattr(m, "tier").value != "glacial"
            ]
            for mem in recent_mems[:3]:
                content = mem.content[:200] if hasattr(mem, "content") else str(mem)[:200]
                tier = mem.tier.value if hasattr(mem, "tier") else "unknown"
                consolidation = getattr(mem, "consolidation_score", 0.5)
                confidence = (
                    "high" if consolidation > 0.7 else "medium" if consolidation > 0.4 else "low"
                )
                context_parts.append(f"- [{tier}|{confidence}] {content}")

            # Format glacial insights separately (long-term patterns)
            glacial_mems = [
                m
                for m in all_memories
                if getattr(m, "tier", None) and getattr(m, "tier").value == "glacial"
            ]
            if glacial_mems:
                context_parts.append("\n[Long-term patterns from previous sessions:]")
                for mem in glacial_mems[:2]:
                    content = mem.content[:250] if hasattr(mem, "content") else str(mem)[:250]
                    consolidation = getattr(mem, "consolidation_score", 0.8)
                    context_parts.append(f"- [glacial|foundational] {content}")

            context = "\n".join(context_parts)
            self._continuum_context_cache[task_hash] = context
            logger.info(
                f"  [continuum] Retrieved {len(recent_mems)} recent + {len(glacial_mems)} glacial "
                f"memories for domain '{domain}'"
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
            task_hash = self._get_task_hash(task)
            existing_pack = self._research_evidence_pack.get(task_hash)
            if existing_pack:
                existing_ids = {s.id for s in existing_pack.snippets}
                new_snippets = [s for s in evidence_pack.snippets if s.id not in existing_ids]
                existing_pack.snippets.extend(new_snippets)
                existing_pack.total_searched += evidence_pack.total_searched
            else:
                self._research_evidence_pack[task_hash] = evidence_pack

            # Store in memory for future debates
            if (
                evidence_pack.snippets
                and evidence_store_callback
                and callable(evidence_store_callback)
            ):
                evidence_store_callback(evidence_pack.snippets, task)

            return len(evidence_pack.snippets), self._research_evidence_pack.get(task_hash)

        except Exception as e:
            logger.warning(f"Evidence refresh failed: {e}")
            return 0, None

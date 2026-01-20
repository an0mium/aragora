"""
Context initialization phase for debate orchestration.

This module extracts the context initialization logic (Phase 0) from the
Arena._run_inner() method, handling:
- Fork debate history injection
- Trending topic context
- Historical context fetching
- Pattern injection from InsightStore
- Memory pattern injection from CritiqueStore
- Pre-debate research
- DebateResult initialization
- Proposer selection
"""

import asyncio
import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)

# Knowledge query cache (TTL-based to reduce redundant semantic searches)
_knowledge_cache: Dict[str, Tuple[str, float]] = {}
_KNOWLEDGE_CACHE_TTL = 300.0  # 5 minutes

# Check for RLM availability (prefer factory for TRUE RLM support)
try:
    from aragora.rlm import get_rlm, RLMConfig, RLMContext as _RLMContext, HAS_OFFICIAL_RLM

    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HAS_OFFICIAL_RLM = False
    get_rlm = None  # type: ignore[misc,assignment]
    RLMConfig = None  # type: ignore[misc,assignment]
    _RLMContext = None  # type: ignore[misc,assignment]


class ContextInitializer:
    """
    Initializes debate context before the proposal phase.

    This class encapsulates all the context preparation logic that was
    previously in the first ~130 lines of Arena._run_inner().

    Usage:
        initializer = ContextInitializer(
            initial_messages=arena.initial_messages,
            trending_topic=arena.trending_topic,
            recorder=arena.recorder,
            debate_embeddings=arena.debate_embeddings,
            insight_store=arena.insight_store,
            memory=arena.memory,
            protocol=arena.protocol,
        )
        await initializer.initialize(ctx)
    """

    def __init__(
        self,
        initial_messages: Optional[list] = None,
        trending_topic: Any = None,
        recorder: Any = None,
        debate_embeddings: Any = None,
        insight_store: Any = None,
        memory: Any = None,
        protocol: Any = None,
        evidence_collector: Any = None,
        dissent_retriever: Any = None,  # DissentRetriever for historical minority views
        pulse_manager: Any = None,  # PulseManager for trending topics
        auto_fetch_trending: bool = False,  # Auto-fetch trending if no topic provided
        # Knowledge Mound integration
        knowledge_mound: Any = None,  # KnowledgeMound for unified knowledge queries
        enable_knowledge_retrieval: bool = True,  # Query mound before debates
        # Belief Network guidance
        enable_belief_guidance: bool = True,  # Inject historical cruxes from similar debates
        # Cross-debate memory for institutional knowledge
        cross_debate_memory: Any = None,  # CrossDebateMemory for institutional knowledge
        enable_cross_debate_memory: bool = True,  # Query cross-debate memory before debates
        # RLM (Recursive Language Models) for context compression
        enable_rlm_compression: bool = True,  # Compress accumulated context hierarchically
        rlm_config: Any = None,  # RLMConfig for compression settings
        rlm_agent_call: Optional[
            Callable[[str, str], str]
        ] = None,  # Agent callback for compression
        # Callbacks for orchestrator methods
        fetch_historical_context: Optional[Callable] = None,
        format_patterns_for_prompt: Optional[Callable] = None,
        get_successful_patterns_from_memory: Optional[Callable] = None,
        perform_research: Optional[Callable] = None,
        fetch_knowledge_context: Optional[Callable] = None,  # Callback to fetch knowledge context
    ):
        """
        Initialize the context initializer.

        Args:
            initial_messages: Fork debate history messages
            trending_topic: Optional trending topic to inject
            recorder: Optional ReplayRecorder
            debate_embeddings: Optional DebateEmbeddings for historical context
            insight_store: Optional InsightStore for pattern injection
            memory: Optional CritiqueStore for memory patterns
            protocol: DebateProtocol configuration
            evidence_collector: Optional EvidenceCollector for auto-collecting evidence
            dissent_retriever: Optional DissentRetriever for historical minority views
            pulse_manager: Optional PulseManager for fetching trending topics
            auto_fetch_trending: If True and no trending_topic provided, auto-fetch from Pulse
            knowledge_mound: Optional KnowledgeMound for unified knowledge queries
            enable_knowledge_retrieval: If True, query mound for relevant knowledge
            fetch_historical_context: Async callback to fetch historical context
            format_patterns_for_prompt: Callback to format patterns for prompts
            get_successful_patterns_from_memory: Callback to get memory patterns
            perform_research: Async callback to perform pre-debate research
            fetch_knowledge_context: Async callback to fetch knowledge from mound
        """
        self.initial_messages = initial_messages or []
        self.trending_topic = trending_topic
        self.recorder = recorder
        self.debate_embeddings = debate_embeddings
        self.insight_store = insight_store
        self.memory = memory
        self.protocol = protocol
        self.evidence_collector = evidence_collector
        self.dissent_retriever = dissent_retriever
        self.pulse_manager = pulse_manager
        self.auto_fetch_trending = auto_fetch_trending
        self.knowledge_mound = knowledge_mound
        self.enable_knowledge_retrieval = enable_knowledge_retrieval
        self.enable_belief_guidance = enable_belief_guidance
        self.cross_debate_memory = cross_debate_memory
        self.enable_cross_debate_memory = enable_cross_debate_memory

        # RLM configuration - use factory for TRUE RLM support
        self.enable_rlm_compression = enable_rlm_compression and HAS_RLM
        self._rlm: Optional[Any] = None
        if self.enable_rlm_compression and get_rlm is not None:
            try:
                config = rlm_config if rlm_config else (RLMConfig() if RLMConfig else None)
                self._rlm = get_rlm(config=config)
                if HAS_OFFICIAL_RLM:
                    logger.info(
                        "[rlm] TRUE RLM enabled for context initialization "
                        "(REPL-based, model writes code to examine context)"
                    )
                else:
                    logger.info(
                        "[rlm] RLM compression enabled for context initialization "
                        "(compression fallback - install rlm for TRUE RLM)"
                    )
            except Exception as e:
                logger.warning(f"[rlm] Failed to initialize AragoraRLM: {e}")

        # Callbacks
        self._fetch_historical_context = fetch_historical_context
        self._format_patterns_for_prompt = format_patterns_for_prompt
        self._get_successful_patterns_from_memory = get_successful_patterns_from_memory
        self._perform_research = perform_research
        self._fetch_knowledge_context = fetch_knowledge_context

    async def initialize(self, ctx: "DebateContext") -> None:
        """
        Initialize the debate context.

        This method performs context preparation with critical items first
        to enable parallel execution with the proposal phase:

        CRITICAL (must complete before proposals):
        1. Initialize DebateResult
        2. Select proposers

        FAST SYNC (run before background tasks):
        3. Inject fork debate history
        4. Start recorder
        5. Initialize context messages

        BACKGROUND (can run parallel with proposals):
        6. Auto-fetch trending topics from Pulse (if enabled)
        7. Inject trending topic context
        8. Fetch historical context
        9. Fetch knowledge mound context (unified knowledge queries)
        10. Inject learned patterns
        11. Inject memory patterns
        12. Inject historical dissents
        13. Perform pre-debate research
        14. Collect evidence (auto-collection)

        Args:
            ctx: The DebateContext to initialize
        """
        from aragora.core import DebateResult

        # === CRITICAL: Must complete before proposals can start ===

        # 1. Initialize DebateResult (needed for message recording)
        ctx.result = DebateResult(
            task=ctx.env.task,
            messages=[],
            critiques=[],
            votes=[],
            dissenting_views=[],
        )

        # 2. Select proposers (needed by proposal phase)
        self._select_proposers(ctx)
        logger.debug(f"proposers_selected count={len(ctx.proposers)}")

        # === FAST SYNC: Quick operations that set up context ===

        # 3. Inject fork debate history
        self._inject_fork_history(ctx)

        # 4. Start recorder
        self._start_recorder()

        # 5. Initialize context messages for fork debates
        self._init_context_messages(ctx)

        # === BACKGROUND: Context enrichment (can run parallel with proposals) ===
        # These operations gather additional context but aren't blocking.
        # Results are injected before round 2 via await_background_context().

        # 6. Auto-fetch trending topics from Pulse if enabled
        if not self.trending_topic and self.auto_fetch_trending:
            await self._inject_pulse_context(ctx)

        # 7. Inject trending topic context (provided or auto-fetched)
        self._inject_trending_topic(ctx)

        # 8. Fetch historical context (async, with timeout)
        await self._fetch_historical(ctx)

        # 9. Fetch knowledge mound context (unified organizational knowledge)
        await self._inject_knowledge_context(ctx)

        # 10. Inject learned patterns from InsightStore (async)
        await self._inject_insight_patterns(ctx)

        # 11. Inject memory patterns from CritiqueStore
        self._inject_memory_patterns(ctx)

        # 12. Inject historical dissents from ConsensusMemory
        self._inject_historical_dissents(ctx)

        # 12b. Inject belief cruxes from similar past debates (belief guidance)
        if self.enable_belief_guidance:
            self._inject_belief_cruxes(ctx)

        # 12c. Inject cross-debate institutional knowledge
        if self.enable_cross_debate_memory:
            await self._inject_cross_debate_context(ctx)

        # 13. Start research in background (non-blocking for fast startup)
        # Research runs in parallel with proposals, results injected before round 2
        if self.protocol and getattr(self.protocol, "enable_research", False):
            ctx.background_research_task = asyncio.create_task(
                self._perform_pre_debate_research(ctx)
            )
            logger.info("background_research_started")

        # 14. Start evidence collection in background (non-blocking)
        if (
            self.evidence_collector
            and self.protocol
            and getattr(self.protocol, "enable_evidence_collection", True)
        ):
            ctx.background_evidence_task = asyncio.create_task(self._collect_evidence(ctx))
            logger.info("background_evidence_started")

        # 15. Compress accumulated context with RLM (if enabled)
        # Uses TRUE RLM when available, compression fallback otherwise
        if self.enable_rlm_compression and self._rlm and ctx.env.context:
            await self._compress_context_with_rlm(ctx)

    def _inject_fork_history(self, ctx: "DebateContext") -> None:
        """Inject fork debate history into partial messages."""
        from aragora.core import Message

        if not self.initial_messages:
            return

        for msg in self.initial_messages:
            if isinstance(msg, Message):
                ctx.partial_messages.append(msg)
            elif isinstance(msg, dict):
                ctx.partial_messages.append(
                    Message(
                        role=msg.get("role", "user"),
                        agent=msg.get("agent", "fork_context"),
                        content=msg.get("content", ""),
                        round=msg.get("round", 0),
                    )
                )

    def _inject_trending_topic(self, ctx: "DebateContext") -> None:
        """Inject trending topic context into environment."""
        if not self.trending_topic:
            return

        try:
            topic_context = (
                "## TRENDING TOPIC\nThis debate was initiated based on trending topic:\n"
            )
            topic_context += f"- **{self.trending_topic.topic}** ({self.trending_topic.platform})\n"

            if hasattr(self.trending_topic, "category") and self.trending_topic.category:
                topic_context += f"- Category: {self.trending_topic.category}\n"

            if hasattr(self.trending_topic, "volume") and self.trending_topic.volume:
                topic_context += f"- Engagement: {self.trending_topic.volume:,}\n"

            if hasattr(self.trending_topic, "to_debate_prompt"):
                topic_context += f"\n{self.trending_topic.to_debate_prompt()}"

            if ctx.env.context:
                ctx.env.context = topic_context + "\n\n" + ctx.env.context
            else:
                ctx.env.context = topic_context
        except Exception as e:
            logger.debug(f"Trending topic injection failed: {e}")

    async def _inject_pulse_context(self, ctx: "DebateContext") -> None:
        """Auto-fetch and inject trending topics from Pulse.

        Fetches trending topics from configured Pulse ingestors and
        selects the most suitable one for debate context enrichment.
        This runs only if auto_fetch_trending is True and no trending_topic
        was explicitly provided.
        """
        if not self.pulse_manager:
            return

        try:
            topics = await asyncio.wait_for(
                self.pulse_manager.get_trending_topics(limit_per_platform=3),
                timeout=5.0,  # Don't delay debate startup
            )

            if not topics:
                return

            # Select best topic for debate
            if hasattr(self.pulse_manager, "select_topic_for_debate"):
                selected = self.pulse_manager.select_topic_for_debate(topics)
            else:
                selected = topics[0] if topics else None

            if selected:
                # Store as trending_topic so _inject_trending_topic can use it
                self.trending_topic = selected
                logger.info(
                    "[pulse] Auto-selected trending topic: %s (%s)",
                    selected.topic,
                    selected.platform,
                )

        except asyncio.TimeoutError:
            logger.warning("[pulse] Trending topic fetch timed out")
        except Exception as e:
            logger.debug(f"[pulse] Trending topic fetch failed: {e}")

    def _start_recorder(self) -> None:
        """Start the replay recorder if provided."""
        if not self.recorder:
            return

        try:
            self.recorder.start()
            self.recorder.record_phase_change("debate_start")
        except Exception as e:
            logger.warning(f"Recorder start error (non-fatal): {e}")

    async def _fetch_historical(self, ctx: "DebateContext") -> None:
        """Fetch historical context for institutional memory."""
        if not self.debate_embeddings or not self._fetch_historical_context:
            return

        try:
            ctx.historical_context_cache = await asyncio.wait_for(
                self._fetch_historical_context(ctx.env.task, limit=2), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Historical context fetch timed out")
            ctx.historical_context_cache = ""
        except Exception as e:
            logger.debug(f"Historical context fetch error: {e}")
            ctx.historical_context_cache = ""

    async def _inject_knowledge_context(self, ctx: "DebateContext") -> None:
        """Fetch and inject knowledge from Knowledge Mound.

        Queries the unified knowledge superstructure for semantically related
        knowledge items that can inform the debate. This provides agents with
        organizational memory and previously learned conclusions.

        Uses TTL-based caching to reduce redundant semantic searches for
        similar tasks within a short time window.
        """
        global _knowledge_cache

        if not self.knowledge_mound or not self.enable_knowledge_retrieval:
            return

        if not self._fetch_knowledge_context:
            return

        try:
            # Generate cache key from task content
            query_hash = hashlib.md5(ctx.env.task.encode()).hexdigest()

            # Check cache first
            cached = self._get_cached_knowledge(query_hash)
            if cached is not None:
                if cached:  # Non-empty cached result
                    if ctx.env.context:
                        ctx.env.context += "\n\n" + cached
                    else:
                        ctx.env.context = cached
                    logger.info(
                        "[knowledge_mound] Used cached knowledge context (%d chars)",
                        len(cached),
                    )
                return

            # Fetch fresh knowledge context
            knowledge_context = await asyncio.wait_for(
                self._fetch_knowledge_context(ctx.env.task, limit=10),
                timeout=10.0,  # 10 second timeout
            )

            # Cache the result (even if empty, to avoid re-fetching)
            _knowledge_cache[query_hash] = (knowledge_context or "", time.time())

            if knowledge_context:
                if ctx.env.context:
                    ctx.env.context += "\n\n" + knowledge_context
                else:
                    ctx.env.context = knowledge_context
                logger.info(
                    "[knowledge_mound] Injected knowledge context into debate (%d chars)",
                    len(knowledge_context),
                )

        except asyncio.TimeoutError:
            logger.warning("[knowledge_mound] Knowledge context fetch timed out")
        except Exception as e:
            logger.debug(f"[knowledge_mound] Knowledge context fetch error: {e}")

    def _get_cached_knowledge(self, query_hash: str) -> Optional[str]:
        """Get cached knowledge context if still valid.

        Returns:
            Cached knowledge string if found and not expired, None otherwise.
        """
        if query_hash in _knowledge_cache:
            result, ts = _knowledge_cache[query_hash]
            if time.time() - ts < _KNOWLEDGE_CACHE_TTL:
                return result
            # Expired - remove from cache
            del _knowledge_cache[query_hash]
        return None

    async def _inject_insight_patterns(self, ctx: "DebateContext") -> None:
        """Inject learned patterns and high-confidence insights from past debates.

        This method now uses the enhanced get_relevant_insights() to find
        domain-specific insights with high confidence scores, in addition to
        common patterns. Applied insight IDs are stored for usage tracking.
        """
        if not self.insight_store:
            return

        try:
            # 1. Inject common patterns (original behavior)
            patterns = await self.insight_store.get_common_patterns(min_occurrences=2, limit=5)
            if patterns and self._format_patterns_for_prompt:
                pattern_context = self._format_patterns_for_prompt(patterns)
                if ctx.env.context:
                    ctx.env.context += "\n\n" + pattern_context
                else:
                    ctx.env.context = pattern_context

            # 2. Inject high-confidence insights as "learned practices" (B2 enhancement)
            domain = getattr(ctx, "domain", None)
            if domain == "general":
                domain = None

            relevant_insights = await self.insight_store.get_relevant_insights(
                domain=domain,
                min_confidence=0.7,
                limit=3,
            )

            if relevant_insights:
                # Format insights as learned practices
                insight_context = "\n\n## LEARNED PRACTICES (from previous debates)\n"
                insight_context += (
                    "The following insights have proven valuable in similar debates:\n"
                )

                for insight in relevant_insights:
                    insight_context += (
                        f"\n• **{insight.title}** (confidence: {insight.confidence:.0%})\n"
                    )
                    if insight.description:
                        insight_context += f"  {insight.description[:200]}\n"

                    # Track applied insight IDs for usage feedback
                    ctx.applied_insight_ids.append(insight.id)

                if ctx.env.context:
                    ctx.env.context += insight_context
                else:
                    ctx.env.context = insight_context.strip()

                logger.info(
                    "[insight] Injected %d learned practices into debate context",
                    len(relevant_insights),
                )

        except Exception as e:
            logger.debug(f"Pattern injection error: {e}")

    def _inject_memory_patterns(self, ctx: "DebateContext") -> None:
        """Inject successful critique patterns from CritiqueStore memory."""
        if not self.memory or not self._get_successful_patterns_from_memory:
            return

        try:
            memory_patterns = self._get_successful_patterns_from_memory(limit=3)
            if memory_patterns:
                if ctx.env.context:
                    ctx.env.context += "\n\n" + memory_patterns
                else:
                    ctx.env.context = memory_patterns
                logger.info("  [memory] Injected successful critique patterns into debate context")
        except Exception as e:
            logger.debug(f"Memory pattern injection error: {e}")

    def _inject_historical_dissents(self, ctx: "DebateContext") -> None:
        """Inject historical dissenting views from similar past debates.

        Uses DissentRetriever to find relevant contrarian perspectives
        from previous debates on similar topics. This helps prevent
        groupthink and surfaces minority viewpoints that may be valuable.
        """
        if not self.dissent_retriever:
            return

        try:
            # Get debate preparation context with similar debates and dissents
            topic = ctx.env.task
            domain = getattr(ctx, "domain", None)
            if domain == "general":
                domain = None

            historical = self.dissent_retriever.get_debate_preparation_context(
                topic=topic,
                domain=domain,
            )

            if not historical or len(historical.strip()) < 50:
                return

            # Inject as context for all phases
            historical_section = f"\n\n{historical}"
            if ctx.env.context:
                ctx.env.context += historical_section
            else:
                ctx.env.context = historical_section.strip()

            logger.info(
                "[consensus_memory] Injected historical dissent context "
                "(%d chars) from similar debates",
                len(historical),
            )

        except Exception as e:
            logger.debug(f"Historical dissent injection error: {e}")

    def _inject_belief_cruxes(self, ctx: "DebateContext") -> None:
        """Inject belief cruxes from similar past debates.

        Retrieves crux claims (key disagreement points) from past debates
        on similar topics and injects them as context. This helps agents
        focus on the most important points of contention early in the debate.

        Uses the DissentRetriever's underlying ConsensusMemory to find
        similar debates and extract their recorded belief_cruxes.
        """
        if not self.dissent_retriever:
            return

        try:
            # Get the underlying ConsensusMemory from DissentRetriever
            consensus_memory = getattr(self.dissent_retriever, "memory", None)
            if not consensus_memory:
                return

            topic = ctx.env.task
            domain = getattr(ctx, "domain", None)
            if domain == "general":
                domain = None

            # Find similar debates
            similar_debates = consensus_memory.find_similar_debates(
                topic=topic,
                domain=domain,
                min_confidence=0.5,
                limit=5,
            )

            if not similar_debates:
                return

            # Extract belief cruxes from similar debates
            all_cruxes: list[str] = []
            for similar in similar_debates:
                consensus = similar.consensus
                # Cruxes are stored in the metadata dict
                if hasattr(consensus, "metadata") and consensus.metadata:
                    cruxes = consensus.metadata.get("belief_cruxes", [])
                    all_cruxes.extend(cruxes[:3])  # Max 3 per debate

                # Also check key_claims as backup
                if hasattr(consensus, "key_claims") and consensus.key_claims:
                    # Add key claims if we don't have enough cruxes
                    if len(all_cruxes) < 5:
                        all_cruxes.extend(consensus.key_claims[:2])

            if not all_cruxes:
                return

            # Deduplicate and limit
            unique_cruxes = list(dict.fromkeys(all_cruxes))[:5]  # Max 5 cruxes

            # Format as context
            crux_context = "\n\n## HISTORICAL CRUXES (key points of debate from similar topics)\n"
            crux_context += (
                "Previous debates on similar topics identified these as critical decision points:\n"
            )
            for i, crux in enumerate(unique_cruxes, 1):
                # Truncate long cruxes
                crux_text = crux[:300] + "..." if len(crux) > 300 else crux
                crux_context += f"\n{i}. {crux_text}"

            crux_context += "\n\nConsider addressing these points explicitly in your arguments."

            # Inject into context
            if ctx.env.context:
                ctx.env.context += crux_context
            else:
                ctx.env.context = crux_context.strip()

            logger.info(
                "[belief_guidance] Injected %d historical cruxes from %d similar debates",
                len(unique_cruxes),
                len(similar_debates),
            )

        except Exception as e:
            logger.debug(f"[belief_guidance] Crux injection error: {e}")

    async def _inject_cross_debate_context(self, ctx: "DebateContext") -> None:
        """Inject institutional knowledge from CrossDebateMemory.

        Queries the cross-debate memory system for relevant context from past
        debates on similar topics. This provides agents with institutional
        knowledge - conclusions, insights, and patterns that the system has
        learned from previous debates.

        This is distinct from historical dissents (which focus on minority views)
        and belief cruxes (which focus on key disagreement points). Cross-debate
        memory provides a broader view of what the system has learned.
        """
        if not self.cross_debate_memory:
            return

        try:
            topic = ctx.env.task

            # Query cross-debate memory for relevant context
            relevant_context = await asyncio.wait_for(
                self.cross_debate_memory.get_relevant_context(task=topic),
                timeout=5.0,  # Quick timeout to avoid blocking
            )

            if not relevant_context or len(relevant_context.strip()) < 50:
                return

            # Inject as institutional knowledge section
            institutional_section = "\n\n## INSTITUTIONAL KNOWLEDGE\n"
            institutional_section += (
                "The following insights are from previous debates on related topics:\n\n"
            )
            institutional_section += relevant_context

            if ctx.env.context:
                ctx.env.context += institutional_section
            else:
                ctx.env.context = institutional_section.strip()

            logger.info(
                "[cross_debate] Injected institutional knowledge (%d chars) from past debates",
                len(relevant_context),
            )

        except asyncio.TimeoutError:
            logger.debug("[cross_debate] Context fetch timed out")
        except Exception as e:
            logger.debug(f"[cross_debate] Context injection error: {e}")

    async def _perform_pre_debate_research(self, ctx: "DebateContext") -> None:
        """Perform pre-debate research if enabled."""
        if not self.protocol or not getattr(self.protocol, "enable_research", False):
            return

        if not self._perform_research:
            return

        try:
            logger.info("research_start phase=research")
            research_context = await self._perform_research(ctx.env.task)
            if research_context:
                logger.info(f"research_complete chars={len(research_context)}")
                ctx.research_context = research_context
                if ctx.env.context:
                    ctx.env.context += "\n\n" + research_context
                else:
                    ctx.env.context = research_context
            else:
                logger.info("research_empty")
        except Exception as e:
            logger.warning(f"research_error error={e}")
            # Continue without research - don't break the debate

    async def _collect_evidence(self, ctx: "DebateContext") -> None:
        """Collect evidence from configured connectors for debate grounding.

        This auto-collects citations and snippets from connectors like:
        - local_docs: Local documentation
        - github: Code and documentation from GitHub
        - web: Web search results

        Evidence is stored in ctx.evidence_pack and injected into env.context.
        """
        if not self.evidence_collector:
            return

        if not self.protocol or not getattr(self.protocol, "enable_evidence_collection", True):
            return

        try:
            logger.info("evidence_collection_start phase=evidence")
            evidence_pack = await asyncio.wait_for(
                self.evidence_collector.collect_evidence(ctx.env.task),
                timeout=15.0,  # 15 second timeout for evidence collection
            )

            if evidence_pack and evidence_pack.snippets:
                ctx.evidence_pack = evidence_pack
                evidence_context = evidence_pack.to_context_string()
                logger.info(
                    f"evidence_collection_complete snippets={len(evidence_pack.snippets)} "
                    f"sources={evidence_pack.total_searched}"
                )

                # Inject evidence into environment context
                if ctx.env.context:
                    ctx.env.context += "\n\n" + evidence_context
                else:
                    ctx.env.context = evidence_context
            else:
                logger.info("evidence_collection_empty")

        except asyncio.TimeoutError:
            logger.warning("evidence_collection_timeout")
        except Exception as e:
            logger.warning(f"evidence_collection_error error={e}")
            # Continue without evidence - don't break the debate

    async def await_background_context(self, ctx: "DebateContext") -> None:
        """Await and cleanup background research/evidence tasks.

        Called before round 2 to ensure research context is available for critiques.
        This method is safe to call multiple times - completed tasks are cleaned up.
        """
        tasks = []
        task_names = []

        if ctx.background_research_task and not ctx.background_research_task.done():
            tasks.append(ctx.background_research_task)
            task_names.append("research")

        if ctx.background_evidence_task and not ctx.background_evidence_task.done():
            tasks.append(ctx.background_evidence_task)
            task_names.append("evidence")

        if not tasks:
            return

        logger.info(f"awaiting_background_context tasks={task_names}")

        try:
            # Wait up to 30s for background tasks to complete
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0,
            )
            logger.info("background_context_complete")
        except asyncio.TimeoutError:
            logger.warning("background_context_timeout")
            # Cancel any still-running tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

        # Clear task references
        ctx.background_research_task = None
        ctx.background_evidence_task = None

    def _init_context_messages(self, ctx: "DebateContext") -> None:
        """Initialize context messages for fork debates."""
        from aragora.core import Message

        if not self.initial_messages:
            return

        for msg in self.initial_messages:
            if isinstance(msg, dict) and "content" in msg:
                ctx.context_messages.append(
                    Message(
                        agent=msg.get("agent", "previous"),
                        content=msg["content"],
                        role=msg.get("role", "assistant"),
                        round=-1,  # Mark as pre-debate context
                    )
                )

        if ctx.context_messages:
            logger.debug(f"fork_context loaded {len(ctx.context_messages)} initial messages")

    def _select_proposers(self, ctx: "DebateContext") -> None:
        """Select proposers from agent list."""
        ctx.proposers = [a for a in ctx.agents if a.role == "proposer"]

        if not ctx.proposers and ctx.agents:
            # Default to first agent if no dedicated proposers
            ctx.proposers = [ctx.agents[0]]

    async def _compress_context_with_rlm(self, ctx: "DebateContext") -> None:
        """
        Compress accumulated context using Recursive Language Models (RLM).

        Uses AragoraRLM which routes to TRUE RLM (REPL-based) when the official
        library is installed, falling back to compression-based approach otherwise.

        Based on the paper "Recursive Language Models" (arXiv:2512.24601),
        this enables agents to efficiently navigate long content by:
        - TRUE RLM: Model writes code to programmatically examine context
        - Fallback: Creates hierarchical summaries for context compression

        This is particularly valuable when context exceeds agent context windows,
        as it maintains semantic fidelity while enabling 100x longer content.
        """
        if not self._rlm:
            return

        try:
            context_content = ctx.env.context or ""
            if len(context_content) < 1000:
                # Skip compression for very short context
                logger.debug("[rlm] Context too short for compression, skipping")
                return

            # Estimate tokens
            estimated_tokens = len(context_content) // 4

            logger.info(
                "[rlm] Compressing context: %d chars (~%d tokens)",
                len(context_content),
                estimated_tokens,
            )

            # Determine source type from context content
            source_type = "text"
            if "## Round" in context_content or "Proposal" in context_content:
                source_type = "debate"
            elif "def " in context_content or "class " in context_content:
                source_type = "code"

            # Compress using AragoraRLM (routes to TRUE RLM if available)
            compression_result = await asyncio.wait_for(
                self._rlm.compress_and_query(
                    query="Create a comprehensive summary preserving key information",
                    content=context_content,
                    source_type=source_type,
                ),
                timeout=30.0,  # 30 second timeout for compression
            )

            # Store summary in context
            if compression_result and compression_result.answer:
                ctx.rlm_compressed_context = compression_result.answer

                # Log which approach was used
                if compression_result.used_true_rlm:
                    logger.info(
                        "[rlm] Context compressed using TRUE RLM "
                        "(model wrote code to examine content)"
                    )
                elif compression_result.used_compression_fallback:
                    logger.info("[rlm] Context compressed using compression fallback")

                # Calculate compression stats
                compressed_tokens = len(compression_result.answer) // 4
                reduction = ((estimated_tokens - compressed_tokens) / estimated_tokens) * 100

                logger.info(
                    "[rlm] Context compressed: %d → %d tokens (%.0f%% reduction)",
                    estimated_tokens,
                    compressed_tokens,
                    reduction,
                )

                # Optionally replace context with summary for agents with small windows
                if hasattr(ctx, "use_compressed_context") and ctx.use_compressed_context:
                    if len(compression_result.answer) < len(context_content):
                        ctx.env.context = (
                            "## COMPRESSED CONTEXT (full context available on request)\n\n"
                            + compression_result.answer
                        )
                    logger.info("[rlm] Replaced context with summary level")

        except asyncio.TimeoutError:
            logger.warning("[rlm] Context compression timed out after 30s")
        except Exception as e:
            logger.warning("[rlm] Context compression failed: %s", e)
            # Continue without compressed context - don't break the debate

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
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Message
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


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
        # Callbacks for orchestrator methods
        fetch_historical_context: Optional[Callable] = None,
        format_patterns_for_prompt: Optional[Callable] = None,
        get_successful_patterns_from_memory: Optional[Callable] = None,
        perform_research: Optional[Callable] = None,
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
            fetch_historical_context: Async callback to fetch historical context
            format_patterns_for_prompt: Callback to format patterns for prompts
            get_successful_patterns_from_memory: Callback to get memory patterns
            perform_research: Async callback to perform pre-debate research
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

        # Callbacks
        self._fetch_historical_context = fetch_historical_context
        self._format_patterns_for_prompt = format_patterns_for_prompt
        self._get_successful_patterns_from_memory = get_successful_patterns_from_memory
        self._perform_research = perform_research

    async def initialize(self, ctx: "DebateContext") -> None:
        """
        Initialize the debate context.

        This method performs all context preparation in order:
        1. Inject fork debate history
        2. Auto-fetch trending topics from Pulse (if enabled)
        3. Inject trending topic context
        4. Start recorder
        5. Fetch historical context
        6. Inject learned patterns
        7. Inject memory patterns
        8. Inject historical dissents
        9. Perform pre-debate research
        10. Collect evidence (auto-collection)
        11. Initialize DebateResult
        12. Initialize context messages
        13. Select proposers

        Args:
            ctx: The DebateContext to initialize
        """
        from aragora.core import DebateResult, Message

        # 1. Inject fork debate history
        self._inject_fork_history(ctx)

        # 2. Auto-fetch trending topics from Pulse if enabled
        if not self.trending_topic and self.auto_fetch_trending:
            await self._inject_pulse_context(ctx)

        # 3. Inject trending topic context (provided or auto-fetched)
        self._inject_trending_topic(ctx)

        # 4. Start recorder
        self._start_recorder()

        # 5. Fetch historical context (async, with timeout)
        await self._fetch_historical(ctx)

        # 6. Inject learned patterns from InsightStore (async)
        await self._inject_insight_patterns(ctx)

        # 7. Inject memory patterns from CritiqueStore
        self._inject_memory_patterns(ctx)

        # 8. Inject historical dissents from ConsensusMemory
        self._inject_historical_dissents(ctx)

        # 9. Perform pre-debate research (async)
        await self._perform_pre_debate_research(ctx)

        # 10. Collect evidence (auto-collection from connectors)
        await self._collect_evidence(ctx)

        # 11. Initialize DebateResult
        ctx.result = DebateResult(
            task=ctx.env.task,
            messages=[],
            critiques=[],
            votes=[],
            dissenting_views=[],
        )

        # 12. Initialize context messages for fork debates
        self._init_context_messages(ctx)

        # 13. Select proposers
        self._select_proposers(ctx)

    def _inject_fork_history(self, ctx: "DebateContext") -> None:
        """Inject fork debate history into partial messages."""
        from aragora.core import Message

        if not self.initial_messages:
            return

        for msg in self.initial_messages:
            if isinstance(msg, Message):
                ctx.partial_messages.append(msg)
            elif isinstance(msg, dict):
                ctx.partial_messages.append(Message(
                    role=msg.get("role", "user"),
                    agent=msg.get("agent", "fork_context"),
                    content=msg.get("content", ""),
                    round=msg.get("round", 0),
                ))

    def _inject_trending_topic(self, ctx: "DebateContext") -> None:
        """Inject trending topic context into environment."""
        if not self.trending_topic:
            return

        try:
            topic_context = "## TRENDING TOPIC\nThis debate was initiated based on trending topic:\n"
            topic_context += f"- **{self.trending_topic.topic}** ({self.trending_topic.platform})\n"

            if hasattr(self.trending_topic, 'category') and self.trending_topic.category:
                topic_context += f"- Category: {self.trending_topic.category}\n"

            if hasattr(self.trending_topic, 'volume') and self.trending_topic.volume:
                topic_context += f"- Engagement: {self.trending_topic.volume:,}\n"

            if hasattr(self.trending_topic, 'to_debate_prompt'):
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
                timeout=5.0  # Don't delay debate startup
            )

            if not topics:
                return

            # Select best topic for debate
            if hasattr(self.pulse_manager, 'select_topic_for_debate'):
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
                self._fetch_historical_context(ctx.env.task, limit=2),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Historical context fetch timed out")
            ctx.historical_context_cache = ""
        except Exception as e:
            logger.debug(f"Historical context fetch error: {e}")
            ctx.historical_context_cache = ""

    async def _inject_insight_patterns(self, ctx: "DebateContext") -> None:
        """Inject learned patterns from past debates."""
        if not self.insight_store:
            return

        try:
            patterns = await self.insight_store.get_common_patterns(
                min_occurrences=2, limit=5
            )
            if patterns and self._format_patterns_for_prompt:
                pattern_context = self._format_patterns_for_prompt(patterns)
                if ctx.env.context:
                    ctx.env.context += "\n\n" + pattern_context
                else:
                    ctx.env.context = pattern_context
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
            domain = getattr(ctx, 'domain', None)
            if domain == 'general':
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

    async def _perform_pre_debate_research(self, ctx: "DebateContext") -> None:
        """Perform pre-debate research if enabled."""
        if not self.protocol or not getattr(self.protocol, 'enable_research', False):
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

        if not self.protocol or not getattr(self.protocol, 'enable_evidence_collection', True):
            return

        try:
            logger.info("evidence_collection_start phase=evidence")
            evidence_pack = await asyncio.wait_for(
                self.evidence_collector.collect_evidence(ctx.env.task),
                timeout=15.0  # 15 second timeout for evidence collection
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

    def _init_context_messages(self, ctx: "DebateContext") -> None:
        """Initialize context messages for fork debates."""
        from aragora.core import Message

        if not self.initial_messages:
            return

        for msg in self.initial_messages:
            if isinstance(msg, dict) and 'content' in msg:
                ctx.context_messages.append(Message(
                    agent=msg.get('agent', 'previous'),
                    content=msg['content'],
                    role=msg.get('role', 'assistant'),
                    round=-1,  # Mark as pre-debate context
                ))

        if ctx.context_messages:
            logger.debug(f"fork_context loaded {len(ctx.context_messages)} initial messages")

    def _select_proposers(self, ctx: "DebateContext") -> None:
        """Select proposers from agent list."""
        ctx.proposers = [a for a in ctx.agents if a.role == "proposer"]

        if not ctx.proposers and ctx.agents:
            # Default to first agent if no dedicated proposers
            ctx.proposers = [ctx.agents[0]]

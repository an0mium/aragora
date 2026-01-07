"""
Multi-agent debate orchestrator.

Implements the propose -> critique -> revise loop with configurable
debate protocols and consensus mechanisms.
"""

import asyncio
import hashlib
import logging
import queue
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Literal, Optional

from aragora.audience.suggestions import cluster_suggestions, format_for_prompt
from aragora.config import USER_EVENT_QUEUE_SIZE
from aragora.core import Agent, Critique, DebateResult, DisagreementReport, Environment, Message, Vote
from aragora.debate.convergence import (
    ConvergenceDetector,
    ConvergenceResult,
    get_similarity_backend,
)
from aragora.debate.memory_manager import MemoryManager
from aragora.debate.optional_imports import OptionalImports
from aragora.debate.prompt_builder import PromptBuilder
from aragora.debate.protocol import CircuitBreaker, DebateProtocol, user_vote_multiplier
from aragora.debate.roles import (
    CognitiveRole,
    RoleAssignment,
    RoleRotationConfig,
    RoleRotator,
    inject_role_into_prompt,
)
from aragora.debate.sanitization import OutputSanitizer
from aragora.debate.security_barrier import SecurityBarrier, TelemetryVerifier
from aragora.server.prometheus import record_debate_completed
from aragora.spectate.stream import SpectatorStream

# Phase classes for orchestrator decomposition
from aragora.debate.phases import (
    ContextInitializer,
    ProposalPhase,
    DebateRoundsPhase,
    ConsensusPhase,
    AnalyticsPhase,
    FeedbackPhase,
    VotingPhase,
)
from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


class Arena:
    """
    Orchestrates multi-agent debates.

    The Arena manages the flow of a debate:
    1. Proposers generate initial proposals
    2. Critics critique each proposal
    3. Proposers revise based on critique
    4. Repeat for configured rounds
    5. Consensus mechanism selects final answer
    """

    def __init__(
        self,
        environment: Environment,
        agents: list[Agent],
        protocol: DebateProtocol = None,
        memory=None,  # CritiqueStore instance
        event_hooks: dict = None,  # Optional hooks for streaming events
        event_emitter=None,  # Optional event emitter for subscribing to user events
        spectator: SpectatorStream = None,  # Optional spectator stream for real-time events
        debate_embeddings=None,  # DebateEmbeddingsDatabase for historical context
        insight_store=None,  # Optional InsightStore for extracting learnings from debates
        recorder=None,  # Optional ReplayRecorder for debate recording
        agent_weights: dict[str, float] | None = None,  # Optional reliability weights from capability probing
        position_tracker=None,  # Optional PositionTracker for truth-grounded personas
        position_ledger=None,  # Optional PositionLedger for grounded personas
        elo_system=None,  # Optional EloSystem for relationship tracking
        persona_manager=None,  # Optional PersonaManager for agent specialization
        dissent_retriever=None,  # Optional DissentRetriever for historical minority views
        flip_detector=None,  # Optional FlipDetector for position reversal detection
        calibration_tracker=None,  # Optional CalibrationTracker for prediction accuracy
        continuum_memory=None,  # Optional ContinuumMemory for cross-debate learning
        relationship_tracker=None,  # Optional RelationshipTracker for agent relationships
        moment_detector=None,  # Optional MomentDetector for significant moments
        loop_id: str = "",  # Loop ID for multi-loop scoping
        strict_loop_scoping: bool = False,  # Drop events without loop_id when True
        circuit_breaker: CircuitBreaker = None,  # Optional CircuitBreaker for agent failure handling
        initial_messages: list = None,  # Optional initial conversation history (for fork debates)
        trending_topic=None,  # Optional TrendingTopic to seed debate context
    ):
        self.env = environment
        self.agents = agents
        self.protocol = protocol or DebateProtocol()
        self.memory = memory
        self.hooks = event_hooks or {}
        self.event_emitter = event_emitter
        self.spectator = spectator or SpectatorStream(enabled=False)
        self.debate_embeddings = debate_embeddings
        self.insight_store = insight_store
        self.recorder = recorder
        self.agent_weights = agent_weights or {}  # Reliability weights from capability probing
        self.position_tracker = position_tracker  # Truth-grounded persona tracking
        self.position_ledger = position_ledger  # Grounded persona position ledger
        self.elo_system = elo_system  # For relationship tracking
        self.persona_manager = persona_manager  # For agent specialization context
        self.dissent_retriever = dissent_retriever  # For historical minority views in debates
        self.flip_detector = flip_detector  # For detecting position reversals
        self.calibration_tracker = calibration_tracker  # For prediction accuracy tracking
        self.continuum_memory = continuum_memory  # For cross-debate learning
        self.relationship_tracker = relationship_tracker  # For agent relationship metrics
        self.moment_detector = moment_detector  # For detecting significant moments

        # Auto-initialize MomentDetector when elo_system available but no detector provided
        if self.moment_detector is None and self.elo_system:
            try:
                from aragora.agents.grounded import MomentDetector as MD
                self.moment_detector = MD(
                    elo_system=self.elo_system,
                    position_ledger=self.position_ledger,
                    relationship_tracker=self.relationship_tracker,
                )
                logger.debug("Auto-initialized MomentDetector for significant moment detection")
            except ImportError:
                pass  # MomentDetector not available
            except Exception as e:
                logger.debug("MomentDetector auto-init failed: %s", e)

        self.loop_id = loop_id  # Loop ID for scoping events
        self.strict_loop_scoping = strict_loop_scoping  # Enforce loop_id on all events
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.initial_messages = initial_messages or []  # Fork debate initial context
        self.trending_topic = trending_topic  # Optional trending topic to seed context

        # ArgumentCartographer for debate graph visualization
        AC = OptionalImports.get_argument_cartographer()
        self.cartographer = AC() if AC else None

        # Auto-upgrade to ELO-ranked judge selection when elo_system is available
        # Only upgrade from default "random" - don't override explicit user choice
        if self.elo_system and self.protocol.judge_selection == "random":
            self.protocol.judge_selection = "elo_ranked"

        # User participation tracking (thread-safe mailbox pattern)
        self._user_event_queue: queue.Queue = queue.Queue(maxsize=USER_EVENT_QUEUE_SIZE)
        # Use deque with maxlen for O(1) bounded queue operations
        self.user_votes: deque[dict] = deque(maxlen=USER_EVENT_QUEUE_SIZE)
        self.user_suggestions: deque[dict] = deque(maxlen=USER_EVENT_QUEUE_SIZE)

        # Cognitive role rotation (Heavy3-inspired)
        self.role_rotator: Optional[RoleRotator] = None
        self.current_role_assignments: dict[str, RoleAssignment] = {}
        if self.protocol.role_rotation:
            config = self.protocol.role_rotation_config or RoleRotationConfig()
            self.role_rotator = RoleRotator(config)

        # Subscribe to user participation events if emitter provided
        if event_emitter:
            from aragora.server.stream import StreamEventType
            event_emitter.subscribe(self._handle_user_event)

        # Assign roles if not already set
        self._assign_roles()

        # Assign initial stances for asymmetric debate
        self._assign_stances(round_num=0)

        # Apply agreement intensity guidance to all agents
        self._apply_agreement_intensity()

        # Initialize convergence detector if enabled
        self.convergence_detector = None
        if self.protocol.convergence_detection:
            self.convergence_detector = ConvergenceDetector(
                convergence_threshold=self.protocol.convergence_threshold,
                divergence_threshold=self.protocol.divergence_threshold,
                min_rounds_before_check=1,
            )

        # Track responses for convergence detection
        self._previous_round_responses: dict[str, str] = {}

        # Cache for historical context (computed once per debate)
        self._historical_context_cache: str = ""

        # Cache for research context (computed once per debate)
        self._research_context_cache: Optional[str] = None

        # Cache for evidence pack (for grounding verdict with citations)
        self._research_evidence_pack = None

        # Cache for continuum memory context (retrieved once per debate)
        self._continuum_context_cache: str = ""
        self._continuum_retrieved_ids: list = []  # Track retrieved memory IDs for outcome updates

        # Cached similarity backend for vote grouping (avoids recreating per call)
        self._similarity_backend = None

        # Voting phase (handles vote grouping, weighted counting, consensus detection)
        self.voting_phase = VotingPhase(
            protocol=self.protocol,
            similarity_backend=None,  # Lazily initialized
        )

        # Cache for debate domain (computed once per debate)
        self._debate_domain_cache: Optional[str] = None

        # Citation extraction (Heavy3-inspired evidence grounding)
        self.citation_extractor = None
        ExtractorClass = OptionalImports.get_citation_extractor()
        if ExtractorClass:
            self.citation_extractor = ExtractorClass()

        # Initialize PromptBuilder for centralized prompt construction
        self.prompt_builder = PromptBuilder(
            protocol=self.protocol,
            env=self.env,
            memory=self.memory,
            continuum_memory=self.continuum_memory,
            dissent_retriever=self.dissent_retriever,
            role_rotator=self.role_rotator,
            persona_manager=self.persona_manager,
            flip_detector=self.flip_detector,
        )

        # Initialize MemoryManager for centralized memory operations
        self.memory_manager = MemoryManager(
            continuum_memory=self.continuum_memory,
            critique_store=self.memory,
            debate_embeddings=self.debate_embeddings,
            domain_extractor=self._extract_debate_domain,
            event_emitter=self.event_emitter,
            spectator=self.spectator,
            loop_id=self.loop_id,
        )

        # =====================================================================
        # Initialize Phase Classes for Orchestrator Decomposition
        # =====================================================================

        # Phase 0: Context Initialization
        self.context_initializer = ContextInitializer(
            initial_messages=self.initial_messages,
            trending_topic=self.trending_topic,
            recorder=self.recorder,
            debate_embeddings=self.debate_embeddings,
            insight_store=self.insight_store,
            memory=self.memory,
            protocol=self.protocol,
            fetch_historical_context=self._fetch_historical_context,
            format_patterns_for_prompt=self._format_patterns_for_prompt,
            get_successful_patterns_from_memory=self._get_successful_patterns_from_memory,
            perform_research=self._perform_research,
        )

        # Phase 1: Initial Proposals
        self.proposal_phase = ProposalPhase(
            circuit_breaker=self.circuit_breaker,
            position_tracker=self.position_tracker,
            position_ledger=self.position_ledger,
            recorder=self.recorder,
            hooks=self.hooks,
            build_proposal_prompt=self._build_proposal_prompt,
            generate_with_agent=self._generate_with_agent,
            with_timeout=self._with_timeout,
            notify_spectator=self._notify_spectator,
            update_role_assignments=self._update_role_assignments,
            record_grounded_position=self._record_grounded_position,
            extract_citation_needs=self._extract_citation_needs,
        )

        # Phase 2: Debate Rounds (critique/revision loop)
        self.debate_rounds_phase = DebateRoundsPhase(
            protocol=self.protocol,
            circuit_breaker=self.circuit_breaker,
            convergence_detector=self.convergence_detector,
            recorder=self.recorder,
            hooks=self.hooks,
            update_role_assignments=self._update_role_assignments,
            assign_stances=self._assign_stances,
            select_critics_for_proposal=self._select_critics_for_proposal,
            critique_with_agent=self._critique_with_agent,
            build_revision_prompt=self._build_revision_prompt,
            generate_with_agent=self._generate_with_agent,
            with_timeout=self._with_timeout,
            notify_spectator=self._notify_spectator,
            record_grounded_position=self._record_grounded_position,
            check_judge_termination=self._check_judge_termination,
            check_early_stopping=self._check_early_stopping,
        )

        # Phase 3: Consensus Resolution
        self.consensus_phase = ConsensusPhase(
            protocol=self.protocol,
            elo_system=self.elo_system,
            memory=self.memory,
            agent_weights=self.agent_weights,
            flip_detector=self.flip_detector,
            position_tracker=self.position_tracker,
            calibration_tracker=self.calibration_tracker,
            recorder=self.recorder,
            hooks=self.hooks,
            user_votes=self.user_votes,
            vote_with_agent=self._vote_with_agent,
            with_timeout=self._with_timeout,
            select_judge=self._select_judge,
            build_judge_prompt=self._build_judge_prompt,
            generate_with_agent=self._generate_with_agent,
            group_similar_votes=self._group_similar_votes,
            get_calibration_weight=self._get_calibration_weight,
            notify_spectator=self._notify_spectator,
            drain_user_events=self._drain_user_events,
            extract_debate_domain=self._extract_debate_domain,
            get_belief_analyzer=OptionalImports.get_belief_analyzer,
            user_vote_multiplier=user_vote_multiplier,
        )

        # Phases 4-6: Analytics
        self.analytics_phase = AnalyticsPhase(
            memory=self.memory,
            insight_store=self.insight_store,
            recorder=self.recorder,
            event_emitter=self.event_emitter,
            hooks=self.hooks,
            loop_id=self.loop_id,
            notify_spectator=self._notify_spectator,
            update_agent_relationships=self._update_agent_relationships,
            generate_disagreement_report=self._generate_disagreement_report,
            create_grounded_verdict=self._create_grounded_verdict,
            verify_claims_formally=self._verify_claims_formally,
            format_conclusion=self._format_conclusion,
        )

        # Phase 7: Feedback Loops
        self.feedback_phase = FeedbackPhase(
            elo_system=self.elo_system,
            persona_manager=self.persona_manager,
            position_ledger=self.position_ledger,
            relationship_tracker=self.relationship_tracker,
            moment_detector=self.moment_detector,
            debate_embeddings=self.debate_embeddings,
            flip_detector=self.flip_detector,
            continuum_memory=self.continuum_memory,
            event_emitter=self.event_emitter,
            loop_id=self.loop_id,
            emit_moment_event=self._emit_moment_event,
            store_debate_outcome_as_memory=self._store_debate_outcome_as_memory,
            update_continuum_memory_outcomes=self._update_continuum_memory_outcomes,
            index_debate_async=self._index_debate_async,
        )

    def _require_agents(self) -> list[Agent]:
        """Return agents list, raising error if empty.

        Use this helper before accessing self.agents[0], self.agents[-1],
        or random.choice(self.agents) to prevent IndexError on empty lists.
        """
        if not self.agents:
            raise ValueError("No agents available - Arena requires at least one agent")
        return self.agents

    def _sync_prompt_builder_state(self) -> None:
        """Sync Arena state to PromptBuilder before building prompts.

        This ensures PromptBuilder has access to dynamic state computed by Arena:
        - Role assignments (updated per round)
        - Historical context cache (computed once per debate)
        - Continuum memory context (computed once per debate)
        - User suggestions (accumulated from audience)
        """
        self.prompt_builder.current_role_assignments = self.current_role_assignments
        self.prompt_builder._historical_context_cache = self._historical_context_cache
        self.prompt_builder._continuum_context_cache = self._get_continuum_context()
        self.prompt_builder.user_suggestions = self.user_suggestions

    def _get_continuum_context(self) -> str:
        """Retrieve relevant memories from ContinuumMemory for debate context.

        Uses the debate task and domain to query for related past learnings.
        Enhanced with tier-aware retrieval and confidence markers.
        """
        if self._continuum_context_cache:
            return self._continuum_context_cache

        if not self.continuum_memory:
            return ""

        try:
            domain = self._extract_debate_domain()
            query = f"{domain}: {self.env.task[:200]}"

            # Retrieve memories, prioritizing fast/medium tiers (skip glacial for speed)
            memories = self.continuum_memory.retrieve(
                query=query,
                limit=5,
                min_importance=0.3,  # Only important memories
            )

            if not memories:
                return ""

            # Track retrieved memory IDs for outcome updates after debate
            self._continuum_retrieved_ids = [
                getattr(mem, 'id', None) for mem in memories if getattr(mem, 'id', None)
            ]

            # Format memories with confidence markers based on consolidation
            context_parts = ["[Previous learnings relevant to this debate:]"]
            for mem in memories[:3]:  # Top 3 most relevant
                content = mem.content[:200] if hasattr(mem, 'content') else str(mem)[:200]
                tier = mem.tier.value if hasattr(mem, 'tier') else "unknown"
                # Consolidation score indicates reliability
                consolidation = getattr(mem, 'consolidation_score', 0.5)
                confidence = "high" if consolidation > 0.7 else "medium" if consolidation > 0.4 else "low"
                context_parts.append(f"- [{tier}|{confidence}] {content}")

            self._continuum_context_cache = "\n".join(context_parts)
            logger.info(f"  [continuum] Retrieved {len(memories)} relevant memories for domain '{domain}'")
            return self._continuum_context_cache
        except Exception as e:
            logger.warning(f"  [continuum] Memory retrieval error: {e}")
            return ""

    def _store_debate_outcome_as_memory(self, result: "DebateResult") -> None:
        """Store debate outcome in ContinuumMemory for future retrieval."""
        self.memory_manager.store_debate_outcome(result, self.env.task)

    def _store_evidence_in_memory(self, evidence_snippets: list, task: str) -> None:
        """Store collected evidence snippets in ContinuumMemory for future retrieval."""
        self.memory_manager.store_evidence(evidence_snippets, task)

    def _update_continuum_memory_outcomes(self, result: "DebateResult") -> None:
        """Update retrieved memories based on debate outcome."""
        # Sync tracked IDs to memory manager
        self.memory_manager.track_retrieved_ids(self._continuum_retrieved_ids)
        self.memory_manager.update_memory_outcomes(result)
        # Clear local tracking
        self._continuum_retrieved_ids = []

    def _extract_citation_needs(self, proposals: dict[str, str]) -> dict[str, list[dict]]:
        """Extract claims that need citations from all proposals.

        Heavy3-inspired: Identifies statements that should be backed by evidence.
        """
        if not self.citation_extractor:
            return {}

        citation_needs = {}
        for agent_name, proposal in proposals.items():
            needs = self.citation_extractor.identify_citation_needs(proposal)
            if needs:
                citation_needs[agent_name] = needs
                # Log high-priority citation needs
                high_priority = [n for n in needs if n["priority"] == "high"]
                if high_priority:
                    logger.debug(f"citations_needed agent={agent_name} count={len(high_priority)}")

        return citation_needs

    def _extract_debate_domain(self) -> str:
        """Extract domain from the debate task for calibration tracking.

        Uses heuristics to categorize the debate topic.
        Result is cached since the task doesn't change during a debate.
        """
        # Return cached domain if available
        if self._debate_domain_cache is not None:
            return self._debate_domain_cache

        task_lower = self.env.task.lower()

        # Domain detection heuristics
        if any(w in task_lower for w in ["security", "hack", "vulnerability", "auth", "encrypt"]):
            domain = "security"
        elif any(w in task_lower for w in ["performance", "speed", "optimize", "cache", "latency"]):
            domain = "performance"
        elif any(w in task_lower for w in ["test", "testing", "coverage", "regression"]):
            domain = "testing"
        elif any(w in task_lower for w in ["design", "architecture", "pattern", "structure"]):
            domain = "architecture"
        elif any(w in task_lower for w in ["bug", "error", "fix", "crash", "exception"]):
            domain = "debugging"
        elif any(w in task_lower for w in ["api", "endpoint", "rest", "graphql"]):
            domain = "api"
        elif any(w in task_lower for w in ["database", "sql", "query", "schema"]):
            domain = "database"
        elif any(w in task_lower for w in ["ui", "frontend", "react", "css", "layout"]):
            domain = "frontend"
        else:
            domain = "general"

        # Cache and return
        self._debate_domain_cache = domain
        return domain

    def _get_calibration_weight(self, agent_name: str) -> float:
        """Get agent weight based on calibration score (0.5-1.5 range).

        Uses calibration_score from ELO system to weight agent contributions.
        Agents with better calibration (more accurate confidence estimates)
        have higher weight in voting and selection decisions.

        Returns:
            Weight between 0.5 (uncalibrated/poor) and 1.5 (perfect calibration)
        """
        if not self.elo_system:
            return 1.0

        try:
            rating = self.elo_system.get_rating(agent_name)
            # calibration_score is 0-1, with 0 for agents with < MIN_COUNT predictions
            cal_score = rating.calibration_score
            # Map 0-1 to 0.5-1.5 range: uncalibrated gets 0.5, perfect gets 1.5
            return 0.5 + cal_score
        except Exception as e:
            logger.debug(f"Calibration weight lookup failed for {agent_name}: {e}")
            return 1.0

    def _compute_composite_judge_score(self, agent_name: str) -> float:
        """Compute composite score for judge selection (ELO + calibration).

        Combines ELO ranking with calibration score for more nuanced judge selection.
        Well-calibrated agents with high ELO make better judges.

        Returns:
            Composite score (higher is better)
        """
        if not self.elo_system:
            return 0.0

        try:
            rating = self.elo_system.get_rating(agent_name)
            # Normalize ELO: 1000 is baseline, 500 is typical deviation
            elo_normalized = (rating.elo - 1000) / 500  # ~-1 to 3 range typically
            elo_normalized = max(0, elo_normalized)  # Floor at 0

            # Calibration score is already 0-1
            cal_score = rating.calibration_score

            # Weighted combination: 70% ELO, 30% calibration
            return (elo_normalized * 0.7) + (cal_score * 0.3)
        except Exception as e:
            logger.debug(f"Composite score calculation failed for {agent_name}: {e}")
            return 0.0

    def _select_critics_for_proposal(self, proposal_agent: str, all_critics: list[Agent]) -> list[Agent]:
        """Select which critics should critique the given proposal based on topology."""
        if self.protocol.topology == "all-to-all":
            # All critics except self
            return [c for c in all_critics if c.name != proposal_agent]

        elif self.protocol.topology == "round-robin":
            # Simple round-robin: each critic critiques the next one in alphabetical order
            # First, filter out the proposer from critics
            eligible_critics = [c for c in all_critics if c.name != proposal_agent]
            if not eligible_critics:
                return []
            # Sort critics by name for deterministic ordering
            eligible_critics_sorted = sorted(eligible_critics, key=lambda c: c.name)
            # Each proposal gets critiqued by the "next" critic based on hash
            # Use stable hash for deterministic critic assignment across Python sessions
            proposal_hash = int(hashlib.sha256(proposal_agent.encode()).hexdigest(), 16)
            proposal_index = proposal_hash % len(eligible_critics_sorted)
            return [eligible_critics_sorted[proposal_index]]

        elif self.protocol.topology == "ring":
            # Ring topology: each agent critiques its "neighbors"
            agent_names = sorted([a.name for a in all_critics] + [proposal_agent])
            agent_names.remove(proposal_agent)
            if not agent_names:
                return []
            # Find position of proposal_agent in the ring
            all_names = sorted([a.name for a in self.agents])
            if proposal_agent not in all_names:
                return all_critics  # fallback
            idx = all_names.index(proposal_agent)
            # Critique by left and right neighbors in the ring
            left = all_names[(idx - 1) % len(all_names)]
            right = all_names[(idx + 1) % len(all_names)]
            return [c for c in all_critics if c.name in (left, right)]

        elif self.protocol.topology == "star":
            # Star topology: hub agent critiques everyone, or everyone critiques hub
            if self.protocol.topology_hub_agent:
                hub = self.protocol.topology_hub_agent
            else:
                # Default hub is first agent
                hub = self._require_agents()[0].name
            if proposal_agent == hub:
                # Hub's proposal gets critiqued by all others
                return [c for c in all_critics if c.name != hub]
            else:
                # Others' proposals get critiqued only by hub (if hub is a critic)
                return [c for c in all_critics if c.name == hub]

        elif self.protocol.topology in ("sparse", "random-graph"):
            # Random subset based on sparsity
            available_critics = [c for c in all_critics if c.name != proposal_agent]
            if not available_critics:
                return []
            num_to_select = max(1, int(len(available_critics) * self.protocol.topology_sparsity))
            # Deterministic random based on proposal_agent for reproducibility (stable hash)
            stable_seed = int(hashlib.sha256(proposal_agent.encode()).hexdigest(), 16) % (2**32)
            random.seed(stable_seed)
            selected = random.sample(available_critics, min(num_to_select, len(available_critics)))
            random.seed()  # Reset seed
            return selected

        else:
            # Default to all-to-all
            return [c for c in all_critics if c.name != proposal_agent]

    def _handle_user_event(self, event) -> None:
        """Handle incoming user participation events (thread-safe).

        Events are enqueued for later processing by _drain_user_events().
        This method may be called from any thread (e.g., WebSocket server).
        """
        from aragora.server.stream import StreamEventType

        # Ignore events from other loops to prevent cross-contamination
        event_loop_id = getattr(event, 'loop_id', None)
        if event_loop_id and event_loop_id != self.loop_id:
            return

        # In strict scoping mode, drop events without a loop_id
        if self.strict_loop_scoping and not event_loop_id:
            return

        # Enqueue for processing (thread-safe)
        if event.type in (StreamEventType.USER_VOTE, StreamEventType.USER_SUGGESTION):
            try:
                self._user_event_queue.put_nowait((event.type, event.data))
            except queue.Full:
                logger.warning(f"User event queue full, dropping {event.type}")

    def _drain_user_events(self) -> None:
        """Drain pending user events from queue into working lists.

        This method should be called at safe points in the debate loop:
        - Before building prompts that include audience suggestions
        - Before vote aggregation that includes user votes

        This is the 'digest' phase of the Stadium Mailbox pattern.
        """
        from aragora.server.stream import StreamEventType

        drained_count = 0
        while True:
            try:
                event_type, event_data = self._user_event_queue.get_nowait()
                if event_type == StreamEventType.USER_VOTE:
                    self.user_votes.append(event_data)  # deque auto-evicts oldest
                elif event_type == StreamEventType.USER_SUGGESTION:
                    self.user_suggestions.append(event_data)  # deque auto-evicts oldest
                drained_count += 1
            except queue.Empty:
                break

        if drained_count > 0:
            self._notify_spectator(
                "audience_drain",
                details=f"Processed {drained_count} audience events",
            )

    def _notify_spectator(self, event_type: str, **kwargs):
        """Helper method to emit spectator events and bridge to WebSocket.

        Emits to both SpectatorStream (console/file) and SyncEventEmitter (WebSocket)
        to provide real-time updates to connected clients.
        """
        if self.spectator:
            self.spectator.emit(event_type, **kwargs)

        # Bridge to WebSocket if event_emitter is available
        if self.event_emitter:
            self._emit_spectator_to_websocket(event_type, **kwargs)

    def _emit_spectator_to_websocket(self, event_type: str, **kwargs):
        """Convert spectator event to StreamEvent and emit to WebSocket clients."""
        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            # Map spectator event types to StreamEventType
            type_mapping = {
                "debate_start": StreamEventType.DEBATE_START,
                "debate_end": StreamEventType.DEBATE_END,
                "round": StreamEventType.ROUND_START,
                "round_start": StreamEventType.ROUND_START,
                "propose": StreamEventType.AGENT_MESSAGE,
                "proposal": StreamEventType.AGENT_MESSAGE,
                "critique": StreamEventType.CRITIQUE,
                "vote": StreamEventType.VOTE,
                "consensus": StreamEventType.CONSENSUS,
                "convergence": StreamEventType.CONSENSUS,
                "judge": StreamEventType.AGENT_MESSAGE,
                "memory_recall": StreamEventType.MEMORY_RECALL,
                "audience_drain": StreamEventType.AUDIENCE_DRAIN,
                "audience_summary": StreamEventType.AUDIENCE_SUMMARY,
                "insight_extracted": StreamEventType.INSIGHT_EXTRACTED,
                # Token streaming events
                "token_start": StreamEventType.TOKEN_START,
                "token_delta": StreamEventType.TOKEN_DELTA,
                "token_end": StreamEventType.TOKEN_END,
            }

            stream_type = type_mapping.get(event_type)
            if not stream_type:
                return  # Skip unmapped event types

            # Build StreamEvent from spectator kwargs
            stream_event = StreamEvent(
                type=stream_type,
                data={
                    "details": kwargs.get("details", ""),
                    "metric": kwargs.get("metric"),
                    "event_source": "spectator",
                },
                round=kwargs.get("round_number", 0),
                agent=kwargs.get("agent", ""),
                loop_id=getattr(self, 'loop_id', ''),
            )
            self.event_emitter.emit(stream_event)
        except Exception as e:
            logger.debug(f"Event emission error (non-fatal): {e}")

        # Update ArgumentCartographer with this event
        self._update_cartographer(event_type, **kwargs)

    def _emit_moment_event(self, moment):
        """Emit a significant moment event to WebSocket clients."""
        if not self.event_emitter:
            return
        try:
            from aragora.server.stream import StreamEvent, StreamEventType
            self.event_emitter.emit(StreamEvent(
                type=StreamEventType.MOMENT_DETECTED,
                data=moment.to_dict(),
                debate_id=self.loop_id or "unknown",
            ))
            logger.debug("Emitted moment event: %s for %s", moment.moment_type, moment.agent_name)
        except Exception as e:
            logger.debug("Failed to emit moment event: %s", e)

    def _update_cartographer(self, event_type: str, **kwargs):
        """Update the ArgumentCartographer graph with debate events."""
        if not self.cartographer:
            return
        try:
            agent = kwargs.get("agent", "")
            details = kwargs.get("details", "")
            round_num = kwargs.get("round_number", 0)

            if event_type in ("propose", "proposal"):
                # Record proposal/revision as a node
                self.cartographer.update_from_message(
                    agent=agent,
                    content=details,
                    role="proposer",
                    round_num=round_num,
                )
            elif event_type == "critique":
                # Extract target from details (format: "Critiqued {target}: ...")
                target = ""
                if "Critiqued " in details:
                    target = details.split("Critiqued ")[1].split(":")[0]
                severity = kwargs.get("metric", 0.5)
                self.cartographer.update_from_critique(
                    critic_agent=agent,
                    target_agent=target,
                    severity=severity if isinstance(severity, (int, float)) else 0.5,
                    round_num=round_num,
                    critique_text=details,
                )
            elif event_type == "vote":
                vote_value = details.split(":")[-1].strip() if ":" in details else details
                self.cartographer.update_from_vote(
                    agent=agent,
                    vote_value=vote_value,
                    round_num=round_num,
                )
            elif event_type == "consensus":
                result = details.split(":")[-1].strip() if ":" in details else details
                self.cartographer.update_from_consensus(
                    result=result,
                    round_num=round_num,
                )
        except Exception as e:
            logger.warning(f"Cartographer error (non-fatal): {e}")

    def _record_grounded_position(
        self, agent_name: str, content: str, debate_id: str, round_num: int,
        confidence: float = 0.7, domain: Optional[str] = None,
    ):
        """Record a position to the grounded persona ledger."""
        if not self.position_ledger:
            return
        try:
            self.position_ledger.record_position(
                agent_name=agent_name, claim=content[:1000], confidence=confidence,
                debate_id=debate_id, round_num=round_num, domain=domain,
            )
        except Exception as e:
            logger.warning(f"Position ledger error (non-fatal): {e}")

    def _update_agent_relationships(self, debate_id: str, participants: list[str], winner: Optional[str], votes: list):
        """Update agent relationships after debate completion.

        Uses batch update for O(1) database connections instead of O(n²) for n participants.
        """
        if not self.elo_system:
            return
        try:
            vote_choices = {v.agent: v.choice for v in votes if hasattr(v, 'agent') and hasattr(v, 'choice')}
            # Build batch of relationship updates
            updates = []
            for i, agent_a in enumerate(participants):
                for agent_b in participants[i + 1:]:
                    agreed = agent_a in vote_choices and agent_b in vote_choices and vote_choices[agent_a] == vote_choices[agent_b]
                    a_win = 1 if winner == agent_a else 0
                    b_win = 1 if winner == agent_b else 0
                    updates.append({
                        "agent_a": agent_a,
                        "agent_b": agent_b,
                        "debate_increment": 1,
                        "agreement_increment": 1 if agreed else 0,
                        "a_win": a_win,
                        "b_win": b_win,
                    })
            # Single transaction for all updates
            self.elo_system.update_relationships_batch(updates)
        except Exception as e:
            logger.warning(f"Relationship update error (non-fatal): {e}")

    def _generate_disagreement_report(
        self,
        votes: list[Vote],
        critiques: list[Critique],
        winner: Optional[str] = None,
    ) -> DisagreementReport:
        """
        Generate a DisagreementReport from debate votes and critiques.

        Inspired by Heavy3.ai: surfaces unanimous critiques (high confidence issues)
        and split opinions (risk areas requiring attention).
        """
        report = DisagreementReport()

        if not votes and not critiques:
            return report

        # 1. Analyze vote alignment
        agent_names = list(set(v.agent for v in votes))
        vote_choices = {v.agent: v.choice for v in votes}

        # Calculate agreement score
        if len(vote_choices) > 1:
            choice_counts = Counter(vote_choices.values())
            most_common_list = choice_counts.most_common(1) if choice_counts else []
            most_common_count = most_common_list[0][1] if most_common_list else 0
            report.agreement_score = most_common_count / len(vote_choices)

        # Calculate per-agent alignment with winner
        if winner:
            for agent, choice in vote_choices.items():
                report.agent_alignment[agent] = 1.0 if choice == winner else 0.0

        # 2. Find unanimous critiques (all critics agree on an issue)
        issue_agents: dict[str, set[str]] = {}  # issue text -> set of agents who raised it
        for critique in critiques:
            for issue in critique.issues:
                issue_key = issue.lower().strip()[:100]  # Normalize for matching
                if issue_key not in issue_agents:
                    issue_agents[issue_key] = set()
                issue_agents[issue_key].add(critique.agent)

        # Unanimous = raised by all critics
        critic_agents = set(c.agent for c in critiques)
        if len(critic_agents) > 1:
            for issue, agents in issue_agents.items():
                if agents == critic_agents:
                    # Find the original full issue text
                    for critique in critiques:
                        for orig_issue in critique.issues:
                            if orig_issue.lower().strip()[:100] == issue:
                                report.unanimous_critiques.append(orig_issue)
                                break
                        if issue in [uc.lower().strip()[:100] for uc in report.unanimous_critiques]:
                            break

        # 3. Find split opinions from votes
        if len(vote_choices) > 1:
            unique_choices = set(vote_choices.values())
            if len(unique_choices) > 1:
                # Group agents by their choice
                choice_to_agents: dict[str, list[str]] = {}
                for agent, choice in vote_choices.items():
                    if choice not in choice_to_agents:
                        choice_to_agents[choice] = []
                    choice_to_agents[choice].append(agent)

                # Create split opinion entries
                sorted_choices = sorted(choice_to_agents.items(), key=lambda x: -len(x[1]))
                if len(sorted_choices) >= 2:
                    majority_choice, majority_agents = sorted_choices[0]
                    for minority_choice, minority_agents in sorted_choices[1:]:
                        report.split_opinions.append((
                            f"Vote split: '{majority_choice[:50]}...' vs '{minority_choice[:50]}...'",
                            majority_agents,
                            minority_agents,
                        ))

        # 4. Identify risk areas from low-confidence votes
        low_confidence_topics = []
        for vote in votes:
            if vote.confidence < 0.6:
                low_confidence_topics.append(
                    f"{vote.agent} has low confidence ({vote.confidence:.0%}) in '{vote.choice[:50]}...'"
                )
        report.risk_areas.extend(low_confidence_topics[:5])

        # 5. Add risk areas from high-severity critiques that weren't addressed
        severe_unaddressed = []
        for critique in critiques:
            if critique.severity >= 0.7:
                # Check if the critique target won (meaning issues may remain)
                if winner and critique.target_agent == winner:
                    severe_unaddressed.append(
                        f"High-severity ({critique.severity:.0%}) critique of winner {critique.target_agent}: "
                        f"{critique.issues[0][:100] if critique.issues else 'various issues'}"
                    )
        report.risk_areas.extend(severe_unaddressed[:3])

        return report

    def _link_evidence_to_claim(self, claim_text: str) -> tuple[list, float]:
        """Link evidence snippets to a claim based on keyword matching.

        Returns:
            Tuple of (list of ScholarlyEvidence, grounding_score)
        """
        from aragora.reasoning.citations import (
            ScholarlyEvidence,
            CitationType,
            CitationQuality,
        )

        if not self._research_evidence_pack or not self._research_evidence_pack.snippets:
            return [], 0.0

        # Extract keywords from claim
        claim_lower = claim_text.lower()
        claim_words = set(claim_lower.split())

        matched_citations = []
        for snippet in self._research_evidence_pack.snippets:
            # Calculate relevance based on keyword overlap
            snippet_words = set(snippet.snippet.lower().split())
            snippet_words.update(set(snippet.title.lower().split()))

            # Check for keyword overlap
            overlap = claim_words.intersection(snippet_words)
            if len(overlap) >= 2:  # At least 2 matching keywords
                relevance = len(overlap) / max(len(claim_words), 1)

                # Determine citation type based on source
                source = snippet.source.lower()
                if "github" in source:
                    citation_type = CitationType.CODE_REPOSITORY
                elif "doc" in source or "local" in source:
                    citation_type = CitationType.DOCUMENTATION
                else:
                    citation_type = CitationType.WEB_PAGE

                # Map reliability score to quality
                if snippet.reliability_score >= 0.8:
                    quality = CitationQuality.AUTHORITATIVE
                elif snippet.reliability_score >= 0.6:
                    quality = CitationQuality.REPUTABLE
                elif snippet.reliability_score >= 0.4:
                    quality = CitationQuality.MIXED
                else:
                    quality = CitationQuality.UNVERIFIED

                evidence = ScholarlyEvidence(
                    id=snippet.id,
                    citation_type=citation_type,
                    title=snippet.title,
                    url=snippet.url,
                    excerpt=snippet.snippet[:500],
                    relevance_score=relevance,
                    quality=quality,
                    claim_id=claim_text[:50],  # Truncated claim as ID
                    metadata=snippet.metadata,
                )
                matched_citations.append(evidence)

        # Sort by relevance and take top 3
        matched_citations.sort(key=lambda e: e.relevance_score, reverse=True)
        top_citations = matched_citations[:3]

        # Calculate grounding score based on evidence quality
        if not top_citations:
            return [], 0.0

        # Average quality score weighted by relevance
        total_weight = sum(e.relevance_score for e in top_citations)
        if total_weight == 0:
            return top_citations, 0.3  # Weak grounding

        quality_scores = {
            CitationQuality.PEER_REVIEWED: 1.0,
            CitationQuality.AUTHORITATIVE: 0.9,
            CitationQuality.REPUTABLE: 0.7,
            CitationQuality.MIXED: 0.5,
            CitationQuality.UNVERIFIED: 0.3,
            CitationQuality.QUESTIONABLE: 0.1,
        }

        weighted_score = sum(
            quality_scores.get(e.quality, 0.3) * e.relevance_score
            for e in top_citations
        ) / total_weight

        return top_citations, weighted_score

    def _create_grounded_verdict(self, result: "DebateResult"):
        """Create a GroundedVerdict for the final answer.

        Heavy3-inspired: Wrap final answers with evidence grounding analysis.
        Identifies claims that should be backed by evidence and links available citations.
        """
        if not self.citation_extractor or not result.final_answer:
            return None

        try:
            # Lazy import to avoid circular dependencies
            from aragora.reasoning.citations import GroundedVerdict, CitedClaim

            # Extract claims from the final answer
            claims_text = self.citation_extractor.extract_claims(result.final_answer)

            if not claims_text:
                # No claims needing citations - return minimal grounded verdict
                return GroundedVerdict(
                    verdict=result.final_answer,
                    confidence=result.confidence,
                    grounding_score=1.0,  # No claims = fully grounded (nothing to cite)
                )

            # Create CitedClaim objects with linked evidence
            cited_claims = []
            all_citations = []
            total_grounding = 0.0

            for claim_text in claims_text:
                # Link evidence to this claim
                citations, claim_grounding = self._link_evidence_to_claim(claim_text)
                all_citations.extend(citations)

                claim = CitedClaim(
                    claim_text=claim_text,
                    confidence=result.confidence,
                    grounding_score=claim_grounding,
                    citations=citations,
                )
                cited_claims.append(claim)
                total_grounding += claim_grounding

            # Calculate overall grounding score
            if cited_claims:
                avg_grounding = total_grounding / len(cited_claims)
            else:
                # Fallback: penalize high claim density
                answer_words = len(result.final_answer.split())
                claim_density = len(claims_text) / max(answer_words / 100, 1)
                avg_grounding = max(0.0, 1.0 - (claim_density * 0.2))

            # Deduplicate citations by ID
            seen_ids = set()
            unique_citations = []
            for citation in all_citations:
                if citation.id not in seen_ids:
                    seen_ids.add(citation.id)
                    unique_citations.append(citation)

            return GroundedVerdict(
                verdict=result.final_answer,
                confidence=result.confidence,
                claims=cited_claims,
                all_citations=unique_citations,
                grounding_score=avg_grounding,
            )

        except Exception as e:
            logger.warning(f"Error creating grounded verdict: {e}")
            return None

    async def _verify_claims_formally(self, result: "DebateResult") -> None:
        """Verify decidable claims using Z3 SMT solver.

        For arithmetic, logic, and constraint claims, attempts formal verification
        to provide machine-verified evidence. Results are stored in the grounded_verdict.
        """
        if not result.grounded_verdict or not result.grounded_verdict.claims:
            return

        try:
            from aragora.verification.formal import get_formal_verification_manager, FormalProofStatus
        except ImportError:
            return  # Formal verification not available

        try:
            manager = get_formal_verification_manager()
            status = manager.status_report()

            if not status.get("any_available"):
                return  # No backends available

            verified_count = 0
            disproven_count = 0

            for claim in result.grounded_verdict.claims[:5]:  # Verify top 5 claims
                try:
                    # Attempt formal verification
                    proof_result = await manager.attempt_formal_verification(
                        claim=claim.claim_text,
                        claim_type="decidable",
                        context=result.final_answer[:500] if result.final_answer else "",
                        timeout_seconds=5.0,
                    )

                    if proof_result and proof_result.status == FormalProofStatus.PROOF_FOUND:
                        claim.grounding_score = 1.0  # Formally verified
                        claim.citations.append({
                            "type": "formal_proof",
                            "prover": proof_result.language.value,
                            "verified": True,
                        })
                        verified_count += 1
                    elif proof_result and proof_result.status == FormalProofStatus.PROOF_FAILED:
                        claim.grounding_score = 0.0  # Disproven
                        claim.citations.append({
                            "type": "formal_proof",
                            "prover": proof_result.language.value,
                            "verified": False,
                            "counterexample": proof_result.proof_text,  # proof_text contains counterexample
                        })
                        disproven_count += 1

                except Exception as e:
                    logger.debug(f"Formal verification failed for claim: {e}")

            if verified_count > 0 or disproven_count > 0:
                logger.info(f"  [formal] Z3 verified {verified_count} claims, disproved {disproven_count}")

        except Exception as e:
            logger.debug(f"Formal verification error: {e}")

    async def _fetch_historical_context(self, task: str, limit: int = 3) -> str:
        """Fetch similar past debates for historical context."""
        return await self.memory_manager.fetch_historical_context(task, limit)

    def _format_patterns_for_prompt(self, patterns: list[dict]) -> str:
        """Format learned patterns as prompt context for agents."""
        return self.memory_manager._format_patterns_for_prompt(patterns)

    def _get_successful_patterns_from_memory(self, limit: int = 5) -> str:
        """Retrieve successful patterns from CritiqueStore memory."""
        return self.memory_manager.get_successful_patterns(limit)

    async def _perform_research(self, task: str) -> str:
        """Perform multi-source research for the debate topic and return formatted context.

        Uses EvidenceCollector with multiple connectors:
        - WebConnector: DuckDuckGo search for general web results
        - GitHubConnector: Code/docs from GitHub repositories
        - LocalDocsConnector: Local documentation files

        Also includes pulse/trending context when available.
        """
        context_parts = []

        # === Aragora-Specific Context ===
        # Auto-inject aragora documentation when debate mentions aragora
        task_lower = task.lower()
        is_aragora_topic = any(kw in task_lower for kw in ["aragora", "multi-agent debate", "nomic loop", "debate framework"])

        if is_aragora_topic:
            try:
                from pathlib import Path
                import os

                # Find project root (where CLAUDE.md and docs/ are)
                project_root = Path(__file__).parent.parent.parent
                docs_dir = project_root / "docs"

                aragora_context_parts = []

                # Read key documentation files
                key_docs = ["FEATURES.md", "ARCHITECTURE.md", "QUICKSTART.md", "STATUS.md"]
                for doc_name in key_docs:
                    doc_path = docs_dir / doc_name
                    if doc_path.exists():
                        try:
                            content = doc_path.read_text()[:3000]  # Limit per file
                            aragora_context_parts.append(f"### {doc_name}\n{content}")
                        except Exception as e:
                            logger.debug(f"Failed to read {doc_name}: {e}")

                # Also include CLAUDE.md for project overview
                claude_md = project_root / "CLAUDE.md"
                if claude_md.exists():
                    try:
                        content = claude_md.read_text()[:2000]
                        aragora_context_parts.insert(0, f"### Project Overview (CLAUDE.md)\n{content}")
                    except (OSError, UnicodeDecodeError):
                        # Skip if file can't be read (permission, encoding issues)
                        pass

                if aragora_context_parts:
                    context_parts.append(
                        "## ARAGORA PROJECT CONTEXT\n"
                        "The following is internal documentation about the Aragora project:\n\n"
                        + "\n\n---\n\n".join(aragora_context_parts[:4])  # Limit to 4 docs
                    )
                    logger.info("Injected Aragora project documentation context")

            except Exception as e:
                logger.debug(f"Failed to load Aragora context: {e}")

        # === Evidence Collection ===
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

            # Add local docs connector for project-specific knowledge
            try:
                from aragora.connectors.local_docs import LocalDocsConnector
                from pathlib import Path

                project_root = Path(__file__).parent.parent.parent
                collector.add_connector("local_docs", LocalDocsConnector(
                    root_path=str(project_root / "docs"),
                    file_types="docs"
                ))
                enabled_connectors.append("local_docs")
            except ImportError:
                pass

            # Collect evidence from all available connectors
            if enabled_connectors:
                evidence_pack = await collector.collect_evidence(task, enabled_connectors=enabled_connectors)

                if evidence_pack.snippets:
                    # Store evidence pack for grounding verdict with citations
                    self._research_evidence_pack = evidence_pack
                    # Update prompt_builder with evidence for per-round citation support
                    if hasattr(self, 'prompt_builder') and self.prompt_builder:
                        self.prompt_builder.set_evidence_pack(evidence_pack)
                    # Store evidence in ContinuumMemory for future debates
                    self._store_evidence_in_memory(evidence_pack.snippets, task)
                    context_parts.append(f"## EVIDENCE CONTEXT\n{evidence_pack.to_context_string()}")

        except Exception as e:
            logger.warning(f"Evidence collection failed: {e}")

        # === Pulse/Trending Context ===
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
                for t in topics[:5]:  # Show top 5 from all platforms
                    trending_context += f"- {t.topic} ({t.platform}, {t.volume:,} engagement, {t.category})\n"
                context_parts.append(trending_context)

        except Exception as e:
            logger.debug(f"Pulse context unavailable: {e}")

        if context_parts:
            return "\n\n".join(context_parts)
        else:
            return "No research context available."

    def _format_conclusion(self, result: "DebateResult") -> str:
        """Format a clear, readable debate conclusion with full context."""
        lines = []
        lines.append("=" * 60)
        lines.append("DEBATE CONCLUSION")
        lines.append("=" * 60)

        # Verdict section
        lines.append("\n## VERDICT")
        if result.consensus_reached:
            lines.append(f"Consensus: YES ({result.confidence:.0%} agreement)")
            if hasattr(result, 'consensus_strength') and result.consensus_strength:
                lines.append(f"Strength: {result.consensus_strength.upper()}")
        else:
            lines.append(f"Consensus: NO (only {result.confidence:.0%} agreement)")

        # Winner (if determined)
        if hasattr(result, 'winner') and result.winner:
            lines.append(f"Winner: {result.winner}")

        # Final answer section
        lines.append("\n## FINAL ANSWER")
        if result.final_answer:
            # Truncate if very long, but show substantial content
            answer_display = result.final_answer[:1000] + "..." if len(result.final_answer) > 1000 else result.final_answer
            lines.append(answer_display)
        else:
            lines.append("No final answer determined.")

        # Vote breakdown (if available)
        if hasattr(result, 'votes') and result.votes:
            lines.append("\n## VOTE BREAKDOWN")
            vote_counts = {}
            for vote in result.votes:
                voter = getattr(vote, 'voter', 'unknown')
                choice = getattr(vote, 'choice', 'abstain')
                vote_counts[voter] = choice
            for voter, choice in vote_counts.items():
                lines.append(f"  - {voter}: {choice}")

        # Dissenting views (if any)
        if hasattr(result, 'dissenting_views') and result.dissenting_views:
            lines.append("\n## DISSENTING VIEWS")
            for i, view in enumerate(result.dissenting_views[:3]):
                view_display = view[:300] + "..." if len(view) > 300 else view
                lines.append(f"  {i+1}. {view_display}")

        # Debate cruxes (key disagreement points)
        if hasattr(result, 'belief_cruxes') and result.belief_cruxes:
            lines.append("\n## KEY CRUXES")
            for crux in result.belief_cruxes[:3]:
                claim = crux.get('claim', 'unknown')[:80]
                uncertainty = crux.get('uncertainty', 0)
                lines.append(f"  - {claim}... (uncertainty: {uncertainty:.2f})")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def _assign_roles(self):
        """Assign roles to agents based on protocol with safety bounds."""
        # If agents already have roles, respect them
        if all(a.role for a in self.agents):
            return

        n_agents = len(self.agents)

        # Safety: Ensure at least 1 critic and 1 synthesizer when we have 3+ agents
        # This prevents all-proposer scenarios that break debate dynamics
        max_proposers = max(1, n_agents - 2) if n_agents >= 3 else 1
        proposers_needed = min(self.protocol.proposer_count, max_proposers)

        for i, agent in enumerate(self.agents):
            if i < proposers_needed:
                agent.role = "proposer"
            elif i == n_agents - 1:
                agent.role = "synthesizer"
            else:
                agent.role = "critic"

        # Log role assignment for debugging
        roles = {a.name: a.role for a in self.agents}
        logger.debug(f"Role assignment: {roles}")

    def _apply_agreement_intensity(self):
        """Apply agreement intensity guidance to all agents' system prompts.

        This modifies each agent's system_prompt to include guidance on how
        much to agree vs disagree with other agents, based on the protocol's
        agreement_intensity setting.
        """
        guidance = self._get_agreement_intensity_guidance()

        for agent in self.agents:
            if agent.system_prompt:
                agent.system_prompt = f"{agent.system_prompt}\n\n{guidance}"
            else:
                agent.system_prompt = guidance

    def _assign_stances(self, round_num: int = 0):
        """Assign debate stances to agents for asymmetric debate.

        Stances: "affirmative" (defend), "negative" (challenge), "neutral" (evaluate)
        If rotate_stances is True, stances rotate each round.
        """
        if not self.protocol.asymmetric_stances:
            return

        stances = ["affirmative", "negative", "neutral"]
        n_agents = len(self.agents)

        for i, agent in enumerate(self.agents):
            # Rotate stance based on round number if enabled
            if self.protocol.rotate_stances:
                stance_idx = (i + round_num) % len(stances)
            else:
                stance_idx = i % len(stances)

            agent.stance = stances[stance_idx]

    def _get_stance_guidance(self, agent) -> str:
        """Generate prompt guidance based on agent's debate stance."""
        if not self.protocol.asymmetric_stances:
            return ""

        if agent.stance == "affirmative":
            return """DEBATE STANCE: AFFIRMATIVE
You are assigned to DEFEND and SUPPORT proposals. Your role is to:
- Find strengths and merits in arguments
- Build upon existing ideas
- Advocate for the proposal's value
- Counter criticisms constructively
Even if you personally disagree, argue the affirmative position."""

        elif agent.stance == "negative":
            return """DEBATE STANCE: NEGATIVE
You are assigned to CHALLENGE and CRITIQUE proposals. Your role is to:
- Identify weaknesses, flaws, and risks
- Play devil's advocate
- Raise objections and counterarguments
- Stress-test the proposal
Even if you personally agree, argue the negative position."""

        else:  # neutral
            return """DEBATE STANCE: NEUTRAL
You are assigned to EVALUATE FAIRLY. Your role is to:
- Weigh arguments from both sides impartially
- Identify the strongest and weakest points
- Seek balanced synthesis
- Judge on merit, not position"""

    async def run(self) -> DebateResult:
        """Run the full debate and return results.

        If timeout_seconds is set in protocol, the debate will be terminated
        after the specified time with partial results.
        """
        if self.protocol.timeout_seconds > 0:
            try:
                # Use wait_for for Python 3.10 compatibility (asyncio.timeout is 3.11+)
                return await asyncio.wait_for(
                    self._run_inner(),
                    timeout=self.protocol.timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"debate_timeout timeout_seconds={self.protocol.timeout_seconds}")
                # Return partial result with timeout indicator
                return DebateResult(
                    task=self.env.task,
                    messages=getattr(self, '_partial_messages', []),
                    critiques=getattr(self, '_partial_critiques', []),
                    votes=[],
                    dissenting_views=[],
                    rounds_used=getattr(self, '_partial_rounds', 0),
                )
        return await self._run_inner()

    async def _run_inner(self) -> DebateResult:
        """Internal debate execution orchestrator.

        This method coordinates the debate phases:
        0. Context Initialization - inject history, patterns, research
        1. Proposals - generate initial proposer responses
        2. Debate Rounds - critique/revision loop
        3. Consensus - voting and resolution
        4-6. Analytics - metrics, insights, verdict
        7. Feedback - ELO, persona, position updates
        """
        import uuid

        # Create shared context for all phases
        ctx = DebateContext(
            env=self.env,
            agents=self.agents,
            start_time=time.time(),
            debate_id=str(uuid.uuid4()),
            domain=self._extract_debate_domain(),
        )

        # Initialize result early for timeout recovery
        ctx.result = DebateResult(
            task=self.env.task,
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )

        try:
            # Phase 0: Context Initialization
            await self.context_initializer.initialize(ctx)

            # Phase 1: Initial Proposals
            await self.proposal_phase.execute(ctx)

            # Phase 2: Debate Rounds (critique/revision loop)
            await self.debate_rounds_phase.execute(ctx)

            # Phase 3: Consensus Resolution
            await self.consensus_phase.execute(ctx)

            # Phases 4-6: Analytics
            await self.analytics_phase.execute(ctx)

            # Phase 7: Feedback Loops
            await self.feedback_phase.execute(ctx)

        except asyncio.TimeoutError:
            # Timeout recovery - use partial results from context
            ctx.result.messages = ctx.partial_messages
            ctx.result.critiques = ctx.partial_critiques
            ctx.result.rounds_used = ctx.partial_rounds
            logger.warning("Debate timed out, returning partial results")

        return ctx.result

    # NOTE: Legacy _run_inner code (1,300+ lines) removed after successful phase integration.
    # The debate execution is now handled by phase classes:
    # - ContextInitializer (Phase 0)
    # - ProposalPhase (Phase 1)
    # - DebateRoundsPhase (Phase 2)
    # - ConsensusPhase (Phase 3)
    # - AnalyticsPhase (Phases 4-6)
    # - FeedbackPhase (Phase 7)

    async def _index_debate_async(self, artifact: dict) -> None:
        """Index debate asynchronously to avoid blocking."""
        try:
            if self.debate_embeddings:
                await self.debate_embeddings.index_debate(artifact)
        except Exception as e:
            logger.debug("Async debate indexing failed: %s", e)

    async def _with_timeout(
        self, coro, agent_name: str, timeout_seconds: float = 90.0
    ):
        """
        Wrap coroutine with per-agent timeout.

        If the agent times out, records a circuit breaker failure and raises TimeoutError.
        This prevents a single stalled agent from blocking the entire debate.
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            self.circuit_breaker.record_failure(agent_name)
            logger.warning(f"Agent {agent_name} timed out after {timeout_seconds}s")
            raise TimeoutError(f"Agent {agent_name} timed out after {timeout_seconds}s")

    async def _generate_with_agent(
        self, agent: Agent, prompt: str, context: list[Message]
    ) -> str:
        """Generate response with an agent, handling errors and sanitizing output.

        Implements "Autonomic layer" - catches all exceptions to keep debate alive.
        """
        try:
            raw_output = await agent.generate(prompt, context)
            return OutputSanitizer.sanitize_agent_output(raw_output, agent.name)
        except asyncio.TimeoutError:
            logger.warning(f"[Autonomic] Agent {agent.name} timed out")
            return f"[System: Agent {agent.name} timed out - skipping this turn]"
        except (ConnectionError, OSError) as e:
            # Network/OS errors - log without full traceback
            logger.warning(f"[Autonomic] Agent {agent.name} connection error: {e}")
            return f"[System: Agent {agent.name} connection failed - skipping this turn]"
        except Exception as e:
            # Autonomic containment: convert crashes to valid responses
            logger.exception(f"[Autonomic] Agent {agent.name} failed: {type(e).__name__}: {e}")
            return f"[System: Agent {agent.name} encountered an error - skipping this turn]"

    async def _critique_with_agent(
        self, agent: Agent, proposal: str, task: str, context: list[Message]
    ) -> Optional[Critique]:
        """Get critique from an agent with autonomic error handling."""
        try:
            return await agent.critique(proposal, task, context)
        except asyncio.TimeoutError:
            logger.warning(f"[Autonomic] Agent {agent.name} critique timed out")
            return None
        except (ConnectionError, OSError) as e:
            logger.warning(f"[Autonomic] Agent {agent.name} critique connection error: {e}")
            return None
        except Exception as e:
            logger.exception(f"[Autonomic] Agent {agent.name} critique failed: {e}")
            return None

    async def _vote_with_agent(
        self, agent: Agent, proposals: dict[str, str], task: str
    ) -> Optional[Vote]:
        """Get vote from an agent with autonomic error handling."""
        try:
            return await agent.vote(proposals, task)
        except asyncio.TimeoutError:
            logger.warning(f"[Autonomic] Agent {agent.name} vote timed out")
            return None
        except (ConnectionError, OSError) as e:
            logger.warning(f"[Autonomic] Agent {agent.name} vote connection error: {e}")
            return None
        except Exception as e:
            logger.exception(f"[Autonomic] Agent {agent.name} vote failed: {e}")
            return None

    def _group_similar_votes(self, votes: list[Vote]) -> dict[str, list[str]]:
        """
        Group semantically similar vote choices.

        This prevents artificial disagreement when agents vote for the
        same thing using different wording (e.g., "Vector DB" vs "Use vector database").

        Delegates to VotingPhase for implementation.

        Returns:
            Dict mapping canonical choice -> list of original choices that map to it
        """
        return self.voting_phase.group_similar_votes(votes)

    async def _check_judge_termination(
        self, round_num: int, proposals: dict[str, str], context: list[Message]
    ) -> tuple[bool, str]:
        """
        Have a judge evaluate if the debate is conclusive.

        Returns:
            Tuple of (should_continue: bool, reason: str)
        """
        if not self.protocol.judge_termination:
            return True, ""

        if round_num < self.protocol.min_rounds_before_judge_check:
            return True, ""

        # Select a judge (use existing method)
        judge = await self._select_judge(proposals, context)

        prompt = f"""You are evaluating whether this multi-agent debate has reached a conclusive state.

Task: {self.env.task[:300]}

After {round_num} rounds of debate, the proposals are:
{chr(10).join(f"- {agent}: {prop[:200]}..." for agent, prop in proposals.items())}

Evaluate:
1. Have the key issues been thoroughly discussed?
2. Are there major unresolved disagreements that more debate could resolve?
3. Would additional rounds likely produce meaningful improvements?

Respond with:
CONCLUSIVE: <yes/no>
REASON: <brief explanation>"""

        try:
            response = await self._generate_with_agent(judge, prompt, context[-5:])
            lines = response.strip().split('\n')

            conclusive = False
            reason = ""

            for line in lines:
                if line.upper().startswith("CONCLUSIVE:"):
                    val = line.split(":", 1)[1].strip().lower()
                    conclusive = val in ("yes", "true", "1")
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            if conclusive:
                logger.info(f"judge_termination judge={judge.name} reason={reason[:100]}")
                # Emit event
                if "on_judge_termination" in self.hooks:
                    self.hooks["on_judge_termination"](judge.name, reason)
                return False, reason

        except Exception as e:
            logger.warning(f"Judge termination check failed: {e}")

        return True, ""

    async def _check_early_stopping(
        self, round_num: int, proposals: dict[str, str], context: list[Message]
    ) -> bool:
        """Check if agents want to stop debate early.

        Returns True if debate should continue, False if it should stop.
        """
        if not self.protocol.early_stopping:
            return True  # Continue

        if round_num < self.protocol.min_rounds_before_early_stop:
            return True  # Continue - haven't met minimum rounds

        # Ask each agent if they think more debate would help
        prompt = f"""After {round_num} round(s) of debate on this task:
Task: {self.env.task[:200]}

Current proposals have been critiqued and revised. Do you think additional debate
rounds would significantly improve the answer quality?

Respond with only: CONTINUE or STOP
- CONTINUE: More debate rounds would help refine the answer
- STOP: The proposals are mature enough, further debate is unlikely to help"""

        stop_votes = 0
        total_votes = 0

        tasks = [self._generate_with_agent(agent, prompt, context[-5:]) for agent in self.agents]
        try:
            # Use wait_for for Python 3.10 compatibility (asyncio.timeout is 3.11+)
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.protocol.round_timeout_seconds
            )
        except asyncio.TimeoutError:
            # Timeout during early stopping check - continue debate (safe default)
            logger.warning(f"Early stopping check timed out after {self.protocol.round_timeout_seconds}s")
            return True

        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                continue
            total_votes += 1
            response = result.strip().upper()
            if "STOP" in response and "CONTINUE" not in response:
                stop_votes += 1

        if total_votes == 0:
            return True  # Continue if voting failed

        stop_ratio = stop_votes / total_votes
        should_stop = stop_ratio >= self.protocol.early_stop_threshold

        if should_stop:
            logger.info(f"early_stopping votes={stop_votes}/{total_votes}")
            # Emit early stop event
            if "on_early_stop" in self.hooks:
                self.hooks["on_early_stop"](round_num, stop_votes, total_votes)

        return not should_stop  # Return True to continue, False to stop

    async def _select_judge(self, proposals: dict[str, str], context: list[Message]) -> Agent:
        """Select judge based on protocol.judge_selection setting."""
        if self.protocol.judge_selection == "last":
            # Legacy behavior - use synthesizer or last agent
            synthesizers = [a for a in self.agents if a.role == "synthesizer"]
            return synthesizers[0] if synthesizers else self._require_agents()[-1]

        elif self.protocol.judge_selection == "random":
            # Random selection from all agents
            return random.choice(self._require_agents())

        elif self.protocol.judge_selection == "voted":
            # Agents vote on who should judge
            return await self._vote_for_judge(proposals, context)

        elif self.protocol.judge_selection == "elo_ranked":
            # Select highest ELO-rated agent as judge
            if not self.elo_system:
                logger.warning("elo_ranked judge selection requires elo_system; falling back to random")
                return random.choice(self._require_agents())

            # Get agent names participating in this debate
            agent_names = [a.name for a in self.agents]

            # Query ELO rankings for these agents
            try:
                leaderboard = self.elo_system.get_leaderboard(limit=len(agent_names))
                # Filter to agents in this debate and find highest rated
                for entry in leaderboard:
                    if entry.get("agent") in agent_names:
                        top_agent_name = entry["agent"]
                        top_elo = entry.get("elo", 1500)
                        judge = next((a for a in self.agents if a.name == top_agent_name), None)
                        if judge:
                            logger.debug(f"Selected {top_agent_name} (ELO: {top_elo}) as judge")
                            return judge
            except Exception as e:
                logger.warning(f"ELO query failed: {e}; falling back to random")

            # Fallback if no ELO data
            return random.choice(self._require_agents())

        elif self.protocol.judge_selection == "calibrated":
            # Select based on composite score (ELO + calibration)
            # Prefers well-calibrated agents with high ELO
            if not self.elo_system:
                logger.warning("calibrated judge selection requires elo_system; falling back to random")
                return random.choice(self._require_agents())

            # Score all agents and pick highest
            agent_scores = []
            for agent in self.agents:
                score = self._compute_composite_judge_score(agent.name)
                agent_scores.append((agent, score))

            if agent_scores:
                # Sort by composite score descending
                agent_scores.sort(key=lambda x: x[1], reverse=True)
                best_agent, best_score = agent_scores[0]
                logger.debug(f"Selected {best_agent.name} (composite: {best_score:.3f}) as judge via calibration")
                return best_agent

            # Fallback if scoring failed
            return random.choice(self._require_agents())

        # Default fallback
        return random.choice(self._require_agents())

    async def _vote_for_judge(self, proposals: dict[str, str], context: list[Message]) -> Agent:
        """Have agents vote on who should be the judge."""
        vote_counts: dict[str, int] = {}

        for agent in self.agents:
            # Each agent votes for who should judge (can't vote for self)
            other_agents = [a for a in self.agents if a.name != agent.name]
            prompt = self._build_judge_vote_prompt(other_agents, proposals)

            try:
                raw_response = await agent.generate(prompt, context)
                response = OutputSanitizer.sanitize_agent_output(raw_response, agent.name)
                # Parse vote from response - look for agent names
                for other in other_agents:
                    if other.name.lower() in response.lower():
                        vote_counts[other.name] = vote_counts.get(other.name, 0) + 1
                        break
            except Exception as e:
                logger.debug(f"Synthesizer vote error for {agent.name}: {e}")

        # Select agent with most votes, random tiebreaker
        if vote_counts:
            max_votes = max(vote_counts.values())
            candidates = [name for name, count in vote_counts.items() if count == max_votes]
            winner_name = random.choice(candidates)
            winner = next((a for a in self.agents if a.name == winner_name), None)
            if winner:
                return winner
            logger.warning(f"vote_for_judge_winner_not_found name={winner_name}")

        # Fallback to random if voting fails or winner not found
        return random.choice(self._require_agents())

    def _build_judge_vote_prompt(self, candidates: list[Agent], proposals: dict[str, str]) -> str:
        """Build prompt for voting on who should judge."""
        return self.prompt_builder.build_judge_vote_prompt(candidates, proposals)

    def _get_agreement_intensity_guidance(self) -> str:
        """Generate prompt guidance based on agreement intensity setting.

        Agreement intensity (0-10) affects how agents approach disagreements:
        - Low (0-3): Adversarial - strongly challenge others' positions
        - Medium (4-6): Balanced - judge arguments on merit
        - High (7-10): Collaborative - seek common ground and synthesis
        """
        intensity = self.protocol.agreement_intensity

        if intensity is None:
            return ""  # No agreement intensity guidance when not set

        if intensity <= 1:
            return """IMPORTANT: You strongly disagree with other agents. Challenge every assumption,
find flaws in every argument, and maintain your original position unless presented
with irrefutable evidence. Be adversarial but constructive."""
        elif intensity <= 3:
            return """IMPORTANT: Approach others' arguments with healthy skepticism. Be critical of
proposals and require strong evidence before changing your position. Point out
weaknesses even if you partially agree."""
        elif intensity <= 6:
            return """Evaluate arguments on their merits. Agree when others make valid points,
disagree when you see genuine flaws. Let the quality of reasoning guide your response."""
        elif intensity <= 8:
            return """Look for common ground with other agents. Acknowledge valid points in others'
arguments and try to build on them. Seek synthesis where possible while maintaining
your own reasoned perspective."""
        else:  # 9-10
            return """Actively seek to incorporate other agents' perspectives. Find value in all
proposals and work toward collaborative synthesis. Prioritize finding agreement
and building on others' ideas."""

    def _format_successful_patterns(self, limit: int = 3) -> str:
        """Format successful critique patterns for prompt injection."""
        if not self.memory:
            return ""
        try:
            patterns = self.memory.retrieve_patterns(min_success=2, limit=limit)
            if not patterns:
                return ""

            lines = ["## SUCCESSFUL PATTERNS (from past debates)"]
            for p in patterns:
                issue_preview = p.issue_text[:100] + "..." if len(p.issue_text) > 100 else p.issue_text
                fix_preview = p.suggestion_text[:80] + "..." if len(p.suggestion_text) > 80 else p.suggestion_text
                lines.append(f"- **{p.issue_type}**: {issue_preview}")
                if fix_preview:
                    lines.append(f"  Fix: {fix_preview} ({p.success_count} successes)")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Successful patterns formatting error: {e}")
            return ""

    def _update_role_assignments(self, round_num: int) -> None:
        """Update cognitive role assignments for the current round."""
        if not self.role_rotator:
            return

        agent_names = [a.name for a in self.agents]
        self.current_role_assignments = self.role_rotator.get_assignments(
            agent_names, round_num, self.protocol.rounds
        )

        if self.current_role_assignments:
            roles_str = ", ".join(
                f"{name}: {assign.role.value}"
                for name, assign in self.current_role_assignments.items()
            )
            logger.debug(f"role_assignments round={round_num} roles={roles_str}")

    def _get_role_context(self, agent: Agent) -> str:
        """Get cognitive role context for an agent in the current round."""
        if not self.role_rotator or agent.name not in self.current_role_assignments:
            return ""

        assignment = self.current_role_assignments[agent.name]
        return self.role_rotator.format_role_context(assignment)

    def _get_persona_context(self, agent: Agent) -> str:
        """Get persona context for agent specialization."""
        if not self.persona_manager:
            return ""

        # Try to get persona from database
        persona = self.persona_manager.get_persona(agent.name)
        if not persona:
            # Try default persona based on agent type (e.g., "claude_proposer" -> "claude")
            agent_type = agent.name.split("_")[0].lower()
            from aragora.agents.personas import DEFAULT_PERSONAS
            if agent_type in DEFAULT_PERSONAS:
                # DEFAULT_PERSONAS contains Persona objects directly
                persona = DEFAULT_PERSONAS[agent_type]
            else:
                return ""

        return persona.to_prompt_context()

    def _get_flip_context(self, agent: Agent) -> str:
        """Get flip/consistency context for agent self-awareness.

        This helps agents be aware of their position history and avoid
        unnecessary flip-flopping while still allowing genuine position changes.
        """
        if not self.flip_detector:
            return ""

        try:
            consistency = self.flip_detector.get_agent_consistency(agent.name)

            # Skip if no position history yet
            if consistency.total_positions == 0:
                return ""

            # Only inject context if there are notable flips
            if consistency.total_flips == 0:
                return ""

            # Build context based on flip patterns
            lines = ["## Position Consistency Note"]

            # Warn about contradictions specifically
            if consistency.contradictions > 0:
                lines.append(
                    f"You have {consistency.contradictions} prior position contradiction(s) on record. "
                    "Consider your stance carefully before arguing against positions you previously held."
                )

            # Note retractions
            if consistency.retractions > 0:
                lines.append(
                    f"You have retracted {consistency.retractions} previous position(s). "
                    "If changing positions again, clearly explain your reasoning."
                )

            # Add overall consistency score
            score = consistency.consistency_score
            if score < 0.7:
                lines.append(
                    f"Your consistency score is {score:.0%}. Prioritize coherent positions."
                )

            # Note domains with instability
            if consistency.domains_with_flips:
                domains = ", ".join(consistency.domains_with_flips[:3])
                lines.append(f"Domains with position changes: {domains}")

            return "\n".join(lines) if len(lines) > 1 else ""

        except Exception as e:
            logger.debug(f"Flip context formatting error: {e}")
            return ""

    def _build_proposal_prompt(self, agent: Agent) -> str:
        """Build the initial proposal prompt."""
        # Drain pending audience events before building prompt
        self._drain_user_events()

        # Sync state to PromptBuilder
        self._sync_prompt_builder_state()

        # Compute audience section (needs spectator callback)
        audience_section = ""
        if (
            self.protocol.audience_injection in ("summary", "inject")
            and self.user_suggestions
        ):
            clusters = cluster_suggestions(self.user_suggestions)
            audience_section = format_for_prompt(clusters)

            # Emit stream event for dashboard
            if self.spectator and clusters:
                self._notify_spectator(
                    "audience_summary",
                    details=f"{sum(c.count for c in clusters)} suggestions in {len(clusters)} clusters",
                    metric=len(clusters),
                )

        return self.prompt_builder.build_proposal_prompt(agent, audience_section)

    def _build_revision_prompt(
        self, agent: Agent, original: str, critiques: list[Critique]
    ) -> str:
        """Build the revision prompt including critiques."""
        # Drain pending audience events before building prompt
        self._drain_user_events()

        # Sync state to PromptBuilder
        self._sync_prompt_builder_state()

        # Compute audience section
        audience_section = ""
        if (
            self.protocol.audience_injection in ("summary", "inject")
            and self.user_suggestions
        ):
            clusters = cluster_suggestions(self.user_suggestions)
            audience_section = format_for_prompt(clusters)

        return self.prompt_builder.build_revision_prompt(
            agent, original, critiques, audience_section
        )

    def _build_judge_prompt(
        self, proposals: dict[str, str], task: str, critiques: list[Critique]
    ) -> str:
        """Build the judge/synthesizer prompt."""
        return self.prompt_builder.build_judge_prompt(proposals, task, critiques)

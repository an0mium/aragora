"""
Arena phase initialization helpers.

Keeps Arena orchestration wiring in a dedicated module to reduce
orchestrator size and improve testability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.debate.context_gatherer import ContextGatherer
from aragora.debate.disagreement import DisagreementReporter
from aragora.debate.memory_manager import MemoryManager
from aragora.debate.optional_imports import OptionalImports
from aragora.debate.phases import (
    AnalyticsPhase,
    ConsensusPhase,
    ContextInitializer,
    DebateRoundsPhase,
    FeedbackPhase,
    ProposalPhase,
    VotingPhase,
)
from aragora.debate.prompt_builder import PromptBuilder
from aragora.debate.protocol import user_vote_multiplier
from aragora.reasoning.claims import fast_extract_claims
from aragora.reasoning.evidence_grounding import EvidenceGrounder
from aragora.debate.phase_executor import PhaseConfig, PhaseExecutor

# Optional genesis import for evolution
try:
    from aragora.genesis.breeding import PopulationManager

    GENESIS_AVAILABLE = True
except ImportError:
    PopulationManager = None  # type: ignore[misc,assignment]
    GENESIS_AVAILABLE = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena


def _create_verify_claims_callback(arena: "Arena"):
    """
    Create a verification callback for the consensus phase.

    The callback extracts claims from proposal text and counts verified ones.
    Uses fast_extract_claims for pattern matching, then attempts formal Z3
    verification for LOGICAL and ARITHMETIC claims.

    Args:
        arena: The Arena instance (for future access to formal verification)

    Returns:
        Async callback: (proposal_text: str, limit: int) -> int (verified count)
    """
    # Lazy import formal verification to avoid circular imports
    _formal_manager = None

    def _track_verification(status: str, time_ms: float) -> None:
        """Track verification metrics (lazy import to avoid circular imports)."""
        try:
            from aragora.server.handlers.metrics import track_verification

            track_verification(status, time_ms)
        except ImportError:
            pass  # Metrics not available

    def _get_formal_manager():
        nonlocal _formal_manager
        if _formal_manager is None:
            try:
                from aragora.verification.formal import (
                    get_formal_verification_manager,
                )

                _formal_manager = get_formal_verification_manager()
            except ImportError:
                _formal_manager = False  # Mark as unavailable
        return _formal_manager if _formal_manager is not False else None

    async def verify_claims(proposal_text: str, limit: int = 2) -> dict:
        """
        Verify claims in proposal text and return verification counts.

        Uses a two-tier verification strategy:
        1. For LOGICAL/ARITHMETIC claims: Attempt formal Z3 verification
        2. Fallback: Use confidence threshold from pattern matching

        Args:
            proposal_text: The proposal text to extract and verify claims from
            limit: Maximum number of claims to verify (for performance)

        Returns:
            Dict with "verified" and "disproven" counts (Phase 11A)
        """
        if not proposal_text:
            return {"verified": 0, "disproven": 0}

        # Extract claims using fast pattern matching
        claims = fast_extract_claims(proposal_text, author="proposal")

        if not claims:
            return {"verified": 0, "disproven": 0}

        # Get formal verification manager (lazy loaded)
        formal_manager = _get_formal_manager()
        z3_available = False
        if formal_manager:
            backends = formal_manager.get_available_backends()
            z3_available = any(b.language.value == "z3_smt" for b in backends)

        verified_count = 0
        disproven_count = 0  # Phase 11A: Track disproven claims
        for claim in claims[:limit]:
            claim_type = claim.get("type", "")
            claim_text = claim.get("text", "")
            confidence = claim.get("confidence", 0.0)

            # Try formal Z3 verification for suitable claim types
            if z3_available and claim_type in ("LOGICAL", "ARITHMETIC", "logical", "arithmetic"):
                try:
                    from aragora.verification.formal import FormalProofStatus

                    result = await formal_manager.attempt_formal_verification(
                        claim_text,
                        claim_type=claim_type,
                        timeout_seconds=5.0,  # Short timeout for debate flow
                    )
                    verification_time = result.proof_search_time_ms or 0.0

                    if result.status == FormalProofStatus.PROOF_FOUND:
                        verified_count += 1
                        _track_verification("z3_verified", verification_time)
                        logger.debug(
                            f"claim_z3_verified type={claim_type} "
                            f"proof_time_ms={verification_time:.1f}"
                        )
                        continue
                    elif result.status == FormalProofStatus.PROOF_FAILED:
                        # Z3 found a counterexample - claim is false (Phase 11A)
                        disproven_count += 1
                        _track_verification("z3_disproved", verification_time)
                        logger.debug(
                            f"claim_z3_disproved type={claim_type} "
                            f"counterexample={result.error_message}"
                        )
                        continue  # Don't also count via confidence fallback
                    elif result.status == FormalProofStatus.TIMEOUT:
                        _track_verification("z3_timeout", verification_time)
                    else:
                        # Translation failed or other status
                        _track_verification("z3_translation_failed", verification_time)
                    # Fall through to confidence check
                except Exception as e:
                    logger.debug(f"Z3 verification failed for claim: {e}")

            # Fallback: Count high-confidence claims as "verified"
            # Threshold: 0.5+ confidence from pattern matching
            if confidence >= 0.5:
                verified_count += 1
                _track_verification("confidence_fallback", 0.0)
                logger.debug(f"claim_verified type={claim_type} confidence={confidence:.2f}")

        return {"verified": verified_count, "disproven": disproven_count}

    return verify_claims


def init_phases(arena: "Arena") -> None:
    """Initialize phase classes for orchestrator decomposition."""
    # Voting phase (handles vote grouping, weighted counting, consensus detection)
    arena.voting_phase = VotingPhase(
        protocol=arena.protocol,
        similarity_backend=None,  # Lazily initialized
    )

    # Citation extraction (Heavy3-inspired evidence grounding)
    arena.citation_extractor = None
    extractor_class = OptionalImports.get_citation_extractor()
    if extractor_class:
        arena.citation_extractor = extractor_class()

    # Evidence grounder for creating grounded verdicts with citations
    arena.evidence_grounder = EvidenceGrounder(
        evidence_pack=None,  # Set during research phase
        citation_extractor=arena.citation_extractor,
    )

    # Initialize PromptBuilder for centralized prompt construction
    arena.prompt_builder = PromptBuilder(
        protocol=arena.protocol,
        env=arena.env,
        memory=arena.memory,
        continuum_memory=arena.continuum_memory,
        dissent_retriever=arena.dissent_retriever,
        role_rotator=arena.role_rotator,
        persona_manager=arena.persona_manager,
        flip_detector=arena.flip_detector,
        calibration_tracker=arena.calibration_tracker,
    )

    # Initialize MemoryManager for centralized memory operations
    arena.memory_manager = MemoryManager(
        continuum_memory=arena.continuum_memory,
        critique_store=arena.memory,
        debate_embeddings=arena.debate_embeddings,
        domain_extractor=arena._extract_debate_domain,
        event_emitter=arena.event_emitter,
        spectator=arena.spectator,
        loop_id=arena.loop_id,
        tier_analytics_tracker=getattr(arena, "tier_analytics_tracker", None),
    )

    # Initialize ContextGatherer for research and evidence collection
    # Includes Knowledge Mound auto-grounding for institutional knowledge
    arena.context_gatherer = ContextGatherer(
        evidence_store_callback=arena._store_evidence_in_memory,
        prompt_builder=arena.prompt_builder,
        enable_knowledge_grounding=getattr(arena, "enable_knowledge_retrieval", True),
        knowledge_mound=getattr(arena, "knowledge_mound", None),
        knowledge_workspace_id=getattr(arena, "loop_id", None) or "debate",
    )

    # Auto-initialize PopulationManager for genome evolution when auto_evolve is enabled
    if arena.auto_evolve and arena.population_manager is None and GENESIS_AVAILABLE:
        try:
            arena.population_manager = PopulationManager()
            logger.info("population_manager auto-initialized for genome evolution")
        except Exception as e:
            logger.warning(f"Failed to initialize PopulationManager: {e}")
            arena.population_manager = None

    # Phase 0: Context Initialization
    arena.context_initializer = ContextInitializer(
        initial_messages=arena.initial_messages,
        trending_topic=arena.trending_topic,
        recorder=arena.recorder,
        debate_embeddings=arena.debate_embeddings,
        insight_store=arena.insight_store,
        memory=arena.memory,
        protocol=arena.protocol,
        evidence_collector=arena.evidence_collector,
        dissent_retriever=arena.dissent_retriever,
        pulse_manager=arena.pulse_manager,
        auto_fetch_trending=arena.auto_fetch_trending,
        knowledge_mound=getattr(arena, "knowledge_mound", None),
        enable_knowledge_retrieval=getattr(arena, "enable_knowledge_retrieval", True),
        cross_debate_memory=getattr(arena, "cross_debate_memory", None),
        enable_cross_debate_memory=getattr(arena, "enable_cross_debate_memory", True),
        fetch_historical_context=arena._fetch_historical_context,
        format_patterns_for_prompt=arena._format_patterns_for_prompt,
        get_successful_patterns_from_memory=arena._get_successful_patterns_from_memory,
        perform_research=arena._perform_research,
        fetch_knowledge_context=arena._fetch_knowledge_context,
    )

    # Phase 1: Initial Proposals
    arena.proposal_phase = ProposalPhase(
        circuit_breaker=arena.circuit_breaker,
        position_tracker=arena.position_tracker,
        position_ledger=arena.position_ledger,
        recorder=arena.recorder,
        hooks=arena.hooks,
        build_proposal_prompt=arena._build_proposal_prompt,
        generate_with_agent=arena.autonomic.generate,
        with_timeout=arena.autonomic.with_timeout,
        notify_spectator=arena._notify_spectator,
        update_role_assignments=arena._update_role_assignments,
        record_grounded_position=arena._record_grounded_position,
        extract_citation_needs=arena._extract_citation_needs,
    )

    # Initialize optional advanced features based on protocol flags
    rhetorical_observer = None
    trickster = None

    # Rhetorical Observer for debate pattern detection (concession, rebuttal, synthesis)
    if getattr(arena.protocol, "enable_rhetorical_observer", False):
        try:
            from aragora.debate.rhetorical_observer import get_rhetorical_observer

            rhetorical_observer = get_rhetorical_observer()
            logger.info("rhetorical_observer enabled for debate pattern detection")
        except ImportError as e:
            logger.debug(f"Rhetorical observer unavailable: {e}")

    # Trickster for hollow consensus detection and echo chamber prevention
    if getattr(arena.protocol, "enable_trickster", False):
        try:
            from aragora.debate.trickster import EvidencePoweredTrickster, TricksterConfig

            trickster_config = TricksterConfig(
                sensitivity=getattr(arena.protocol, "trickster_sensitivity", 0.7)
            )
            trickster = EvidencePoweredTrickster(config=trickster_config)
            logger.info("trickster enabled for hollow consensus detection")
        except ImportError as e:
            logger.debug(f"Trickster unavailable: {e}")

    # NoveltyTracker for semantic novelty detection (triggers trickster on staleness)
    # Enabled when trickster is enabled since they work together
    novelty_tracker = None
    if getattr(arena.protocol, "enable_trickster", False):
        try:
            from aragora.debate.novelty import NoveltyTracker

            novelty_tracker = NoveltyTracker(
                low_novelty_threshold=getattr(arena.protocol, "novelty_threshold", 0.15)
            )
            logger.info("novelty_tracker enabled for proposal staleness detection")
        except ImportError as e:
            logger.debug(f"NoveltyTracker unavailable: {e}")

    # Phase 2: Debate Rounds (critique/revision loop)
    arena.debate_rounds_phase = DebateRoundsPhase(
        protocol=arena.protocol,
        circuit_breaker=arena.circuit_breaker,
        convergence_detector=arena.convergence_detector,
        recorder=arena.recorder,
        hooks=arena.hooks,
        trickster=trickster,
        rhetorical_observer=rhetorical_observer,
        event_emitter=arena.event_emitter,
        novelty_tracker=novelty_tracker,
        update_role_assignments=arena._update_role_assignments,
        assign_stances=arena._assign_stances,
        select_critics_for_proposal=arena._select_critics_for_proposal,
        critique_with_agent=arena.autonomic.critique,
        build_revision_prompt=arena._build_revision_prompt,
        generate_with_agent=arena.autonomic.generate,
        with_timeout=arena.autonomic.with_timeout,
        notify_spectator=arena._notify_spectator,
        record_grounded_position=arena._record_grounded_position,
        check_judge_termination=arena._check_judge_termination,
        check_early_stopping=arena._check_early_stopping,
        refresh_evidence=arena._refresh_evidence_for_round,
        checkpoint_callback=arena._create_checkpoint if arena.checkpoint_manager else None,
        context_initializer=arena.context_initializer,
        # RLM compression for long debates (auto-compresses context after threshold rounds)
        compress_context=arena.compress_debate_messages if arena.use_rlm_limiter else None,
        rlm_compression_round_threshold=getattr(arena, "rlm_compression_round_threshold", 3),
        # Adaptive rounds based on memory-aware debate strategy
        debate_strategy=getattr(arena, "debate_strategy", None),
    )

    # Phase 3: Consensus Resolution
    arena.consensus_phase = ConsensusPhase(
        protocol=arena.protocol,
        elo_system=arena.elo_system,
        memory=arena.memory,
        agent_weights=arena.agent_weights,
        flip_detector=arena.flip_detector,
        position_tracker=arena.position_tracker,
        calibration_tracker=arena.calibration_tracker,
        recorder=arena.recorder,
        hooks=arena.hooks,
        user_votes=arena.user_votes,  # type: ignore[arg-type]
        vote_with_agent=arena.autonomic.vote,
        with_timeout=arena.autonomic.with_timeout,
        select_judge=arena._select_judge,
        build_judge_prompt=arena.prompt_builder.build_judge_prompt,
        generate_with_agent=arena.autonomic.generate,
        group_similar_votes=arena._group_similar_votes,
        get_calibration_weight=arena._get_calibration_weight,
        notify_spectator=arena._notify_spectator,
        drain_user_events=arena._drain_user_events,
        extract_debate_domain=arena._extract_debate_domain,
        get_belief_analyzer=OptionalImports.get_belief_analyzer,
        user_vote_multiplier=user_vote_multiplier,
        # Verification callback for claim verification during consensus
        # When protocol.verify_claims_during_consensus is True, this callback
        # extracts claims from proposals and counts verified ones for vote bonuses.
        verify_claims=_create_verify_claims_callback(arena),
    )

    # Phases 4-6: Analytics
    arena.analytics_phase = AnalyticsPhase(
        memory=arena.memory,
        insight_store=arena.insight_store,
        recorder=arena.recorder,
        event_emitter=arena.event_emitter,
        hooks=arena.hooks,
        loop_id=arena.loop_id,
        notify_spectator=arena._notify_spectator,
        update_agent_relationships=arena._update_agent_relationships,
        generate_disagreement_report=lambda votes, critiques, winner=None: DisagreementReporter().generate_report(
            votes, critiques, winner
        ),
        create_grounded_verdict=arena._create_grounded_verdict,
        verify_claims_formally=arena._verify_claims_formally,
        format_conclusion=arena._format_conclusion,
    )

    # Phase 7: Feedback Loops
    arena.feedback_phase = FeedbackPhase(
        elo_system=arena.elo_system,
        persona_manager=arena.persona_manager,
        position_ledger=arena.position_ledger,
        relationship_tracker=arena.relationship_tracker,
        moment_detector=arena.moment_detector,
        debate_embeddings=arena.debate_embeddings,
        flip_detector=arena.flip_detector,
        continuum_memory=arena.continuum_memory,
        event_emitter=arena.event_emitter,
        loop_id=arena.loop_id,
        emit_moment_event=arena._emit_moment_event,
        store_debate_outcome_as_memory=arena._store_debate_outcome_as_memory,
        update_continuum_memory_outcomes=arena._update_continuum_memory_outcomes,
        index_debate_async=arena._index_debate_async,
        consensus_memory=arena.consensus_memory,
        calibration_tracker=arena.calibration_tracker,
        population_manager=arena.population_manager,
        auto_evolve=arena.auto_evolve,
        breeding_threshold=arena.breeding_threshold,
        prompt_evolver=arena.prompt_evolver,
        broadcast_pipeline=arena.extensions.broadcast_pipeline,
        auto_broadcast=arena.extensions.auto_broadcast,
        broadcast_min_confidence=arena.extensions.broadcast_min_confidence,
        knowledge_mound=getattr(arena, "knowledge_mound", None),
        enable_knowledge_ingestion=getattr(arena, "enable_knowledge_ingestion", True),
        ingest_debate_outcome=arena._ingest_debate_outcome,
        knowledge_bridge_hub=getattr(arena, "knowledge_bridge_hub", None),
        # Post-debate workflow automation
        post_debate_workflow=getattr(arena, "post_debate_workflow", None),
        enable_post_debate_workflow=getattr(arena, "enable_post_debate_workflow", False),
        post_debate_workflow_threshold=getattr(arena, "post_debate_workflow_threshold", 0.7),
        # Memory Coordination (atomic cross-system writes)
        memory_coordinator=getattr(arena, "memory_coordinator", None),
        enable_coordinated_writes=getattr(arena, "enable_coordinated_writes", True),
        coordinator_options=getattr(arena, "coordinator_options", None),
        # Selection Feedback Loop (performance → selection)
        selection_feedback_loop=getattr(arena, "selection_feedback_loop", None),
        enable_performance_feedback=getattr(arena, "enable_performance_feedback", True),
    )


def create_phase_executor(arena: "Arena") -> PhaseExecutor:
    """Create and configure the PhaseExecutor for debate execution.

    Calculates dynamic timeout based on agents and rounds, then creates
    the PhaseExecutor with all debate phases.

    Args:
        arena: The Arena instance with initialized phases

    Returns:
        Configured PhaseExecutor instance
    """
    from aragora.config import AGENT_TIMEOUT_SECONDS, MAX_CONCURRENT_CRITIQUES
    from typing import Any

    # Calculate dynamic timeout based on agents and rounds
    # This prevents timeout issues with slow agents (e.g., kimi taking 30-85s/critique)
    num_agents = len(arena.agents) if arena.agents else 4
    num_rounds = getattr(arena.protocol, "rounds", 3)

    # Minimum time: (agents / concurrent) × timeout × 2 phases × rounds + buffer
    min_timeout_needed = (
        (num_agents / max(MAX_CONCURRENT_CRITIQUES, 1))
        * AGENT_TIMEOUT_SECONDS
        * 2  # critique + revision phases per round
        * num_rounds
        + 180  # 3 minute buffer for overhead
    )

    # Use configured timeout if larger, else use calculated minimum (at least 10 minutes)
    base_timeout = getattr(arena.protocol, "timeout", 300.0)
    timeout = max(base_timeout, min_timeout_needed, 600.0)

    logger.info(
        f"debate_timeout_calculated agents={num_agents} rounds={num_rounds} "
        f"base={base_timeout}s calculated={min_timeout_needed:.0f}s final={timeout:.0f}s"
    )

    # Create wrapper for context_initializer to match Phase protocol
    # (context_initializer uses .initialize() instead of .execute())
    class ContextInitWrapper:
        name = "context_initializer"

        def __init__(self, initializer):
            self._initializer = initializer

        async def execute(self, context):
            return await self._initializer.initialize(context)

    phases_dict: dict[str, Any] = {
        "context_initializer": ContextInitWrapper(arena.context_initializer),
        "proposal": arena.proposal_phase,
        "debate_rounds": arena.debate_rounds_phase,
        "consensus": arena.consensus_phase,
        "analytics": arena.analytics_phase,
        "feedback": arena.feedback_phase,
    }

    return PhaseExecutor(
        phases=phases_dict,
        config=PhaseConfig(
            total_timeout_seconds=timeout,
            # Per-phase timeout: 90% of total or at least 300s
            # Higher percentage (was 80%) ensures slow agents complete within phase budget
            phase_timeout_seconds=max(300.0, timeout * 0.9),
            enable_tracing=True,
        ),
    )

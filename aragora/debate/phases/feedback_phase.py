"""
Feedback phase for debate orchestration.

This module extracts the feedback loops logic (Phase 7) from the
Arena._run_inner() method, handling post-debate updates:
1. ELO match recording
2. Persona performance updates
3. Position ledger resolution
4. Relationship tracking
5. Moment detection
6. Debate embedding indexing
7. Flip detection
8. Continuum memory storage
9. Memory outcome updates
10. Calibration data recording
11. Genome fitness updates (Genesis)
12. Population evolution
13. Pulse outcome recording
14. Memory cleanup
15. Evolution pattern extraction
16. Risk assessment
17. Insight usage recording
18. Consensus outcome storage
19. Crux extraction
20. Training data emission (Tinker integration)
21. Coordinated memory writes (cross-system atomic)
22. Selection feedback loop (performance → selection)
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from aragora.agents.errors import _build_error_action
from aragora.debate.phases.consensus_storage import ConsensusStorage
from aragora.debate.phases.training_emitter import TrainingEmitter
from aragora.type_protocols import (
    BroadcastPipelineProtocol,
    CalibrationTrackerProtocol,
    ConsensusMemoryProtocol,
    DebateEmbeddingsProtocol,
    EloSystemProtocol,
    EventEmitterProtocol,
    FlipDetectorProtocol,
    InsightStoreProtocol,
    MomentDetectorProtocol,
    PersonaManagerProtocol,
    PopulationManagerProtocol,
    PositionLedgerProtocol,
    PromptEvolverProtocol,
    PulseManagerProtocol,
    RelationshipTrackerProtocol,
    TieredMemoryProtocol,
)

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.context import DebateContext
    from aragora.memory.consensus import ConsensusStrength

logger = logging.getLogger(__name__)


class FeedbackPhase:
    """
    Executes post-debate feedback loops.

    This class encapsulates all the feedback logic that was previously
    in the final ~200 lines of Arena._run_inner().

    Usage:
        feedback = FeedbackPhase(
            elo_system=arena.elo_system,
            persona_manager=arena.persona_manager,
            position_ledger=arena.position_ledger,
            relationship_tracker=arena.relationship_tracker,
            moment_detector=arena.moment_detector,
            debate_embeddings=arena.debate_embeddings,
            flip_detector=arena.flip_detector,
            continuum_memory=arena.continuum_memory,
            event_emitter=arena.event_emitter,
        )
        await feedback.execute(ctx)
    """

    def __init__(
        self,
        elo_system: Optional[EloSystemProtocol] = None,
        persona_manager: Optional[PersonaManagerProtocol] = None,
        position_ledger: Optional[PositionLedgerProtocol] = None,
        relationship_tracker: Optional[RelationshipTrackerProtocol] = None,
        moment_detector: Optional[MomentDetectorProtocol] = None,
        debate_embeddings: Optional[DebateEmbeddingsProtocol] = None,
        flip_detector: Optional[FlipDetectorProtocol] = None,
        continuum_memory: Optional[TieredMemoryProtocol] = None,
        event_emitter: Optional[EventEmitterProtocol] = None,
        loop_id: Optional[str] = None,
        # Callbacks for orchestrator methods
        emit_moment_event: Optional[Callable[[Any], None]] = None,
        store_debate_outcome_as_memory: Optional[Callable[[Any], None]] = None,
        update_continuum_memory_outcomes: Optional[Callable[[Any], None]] = None,
        index_debate_async: Optional[Callable[[Dict[str, Any]], Any]] = None,
        # ConsensusMemory for storing historical outcomes
        consensus_memory: Optional[ConsensusMemoryProtocol] = None,
        # CalibrationTracker for prediction accuracy
        calibration_tracker: Optional[CalibrationTrackerProtocol] = None,
        # Genesis evolution
        population_manager: Optional[PopulationManagerProtocol] = None,
        auto_evolve: bool = True,
        breeding_threshold: float = 0.8,
        # Pulse manager for trending topic analytics
        pulse_manager: Optional[PulseManagerProtocol] = None,
        # Prompt evolution for learning from debates
        prompt_evolver: Optional[PromptEvolverProtocol] = None,
        # Insight store for tracking applied insights
        insight_store: Optional[InsightStoreProtocol] = None,
        # Training data export for Tinker integration
        training_exporter: Optional[Callable[[List[Dict[str, Any]], str], Any]] = None,
        # Broadcast auto-trigger for high-quality debates
        broadcast_pipeline: Optional[BroadcastPipelineProtocol] = None,
        auto_broadcast: bool = False,
        broadcast_min_confidence: float = 0.8,
        # Knowledge Mound integration
        knowledge_mound: Optional[Any] = None,  # KnowledgeMound for unified knowledge ingestion
        enable_knowledge_ingestion: bool = True,  # Store debate outcomes in mound
        ingest_debate_outcome: Optional[Callable[[Any], Any]] = None,  # Callback to ingest outcome
        knowledge_bridge_hub: Optional[Any] = None,  # KnowledgeBridgeHub for bridge access
        # Memory Coordination (cross-system atomic writes)
        memory_coordinator: Optional[Any] = None,  # MemoryCoordinator for atomic writes
        enable_coordinated_writes: bool = True,  # Use coordinator instead of individual writes
        coordinator_options: Optional[Any] = None,  # CoordinatorOptions for behavior
        # Selection Feedback Loop
        selection_feedback_loop: Optional[Any] = None,  # SelectionFeedbackLoop for performance feedback
        enable_performance_feedback: bool = True,  # Update selection weights based on performance
        # Post-debate workflow automation
        post_debate_workflow: Optional[Any] = None,  # Workflow DAG to trigger after debates
        enable_post_debate_workflow: bool = False,  # Auto-trigger workflow after debates
        post_debate_workflow_threshold: float = 0.7,  # Min confidence to trigger workflow
    ):
        """
        Initialize the feedback phase.

        Args:
            elo_system: Optional ELOSystem for ranking updates
            persona_manager: Optional PersonaManager for performance tracking
            position_ledger: Optional PositionLedger for position resolution
            relationship_tracker: Optional RelationshipTracker
            moment_detector: Optional MomentDetector for narrative moments
            debate_embeddings: Optional DebateEmbeddings for indexing
            flip_detector: Optional FlipDetector for position flips
            continuum_memory: Optional ContinuumMemory for learning
            event_emitter: Optional EventEmitter for WebSocket events
            loop_id: Optional loop ID for event correlation
            emit_moment_event: Callback to emit moment events
            store_debate_outcome_as_memory: Callback to store debate outcome
            update_continuum_memory_outcomes: Callback to update memory outcomes
            index_debate_async: Async callback to index debate
            consensus_memory: Optional ConsensusMemory for storing historical outcomes
            calibration_tracker: Optional CalibrationTracker for prediction accuracy
            population_manager: Optional PopulationManager for genome fitness tracking
            auto_evolve: If True, trigger evolution after high-confidence debates
            breeding_threshold: Minimum confidence to trigger evolution (default 0.8)
            pulse_manager: Optional PulseManager for trending topic analytics
            prompt_evolver: Optional PromptEvolver for extracting winning patterns
            insight_store: Optional InsightStore for insight usage tracking
            training_exporter: Optional callback for exporting training data to Tinker
            broadcast_pipeline: Optional BroadcastPipeline for audio/video generation
            auto_broadcast: If True, trigger broadcast after high-quality debates
            broadcast_min_confidence: Minimum confidence to trigger broadcast (default 0.8)
            knowledge_mound: Optional KnowledgeMound for unified knowledge ingestion
            enable_knowledge_ingestion: If True, store debate outcomes in mound
            ingest_debate_outcome: Async callback to ingest outcome into mound
            knowledge_bridge_hub: Optional KnowledgeBridgeHub for unified bridge access
        """
        self.elo_system = elo_system
        self.persona_manager = persona_manager
        self.position_ledger = position_ledger
        self.relationship_tracker = relationship_tracker
        self.moment_detector = moment_detector
        self.debate_embeddings = debate_embeddings
        self.flip_detector = flip_detector
        self.continuum_memory = continuum_memory
        self.event_emitter = event_emitter
        self.loop_id = loop_id
        self.consensus_memory = consensus_memory
        self.calibration_tracker = calibration_tracker
        self.population_manager = population_manager
        self.auto_evolve = auto_evolve
        self.breeding_threshold = breeding_threshold
        self.pulse_manager = pulse_manager
        self.prompt_evolver = prompt_evolver
        self.insight_store = insight_store
        self.training_exporter = training_exporter
        self.broadcast_pipeline = broadcast_pipeline
        self.auto_broadcast = auto_broadcast
        self.broadcast_min_confidence = broadcast_min_confidence
        self.knowledge_mound = knowledge_mound
        self.enable_knowledge_ingestion = enable_knowledge_ingestion
        self.knowledge_bridge_hub = knowledge_bridge_hub

        # Memory Coordination
        self.memory_coordinator = memory_coordinator
        self.enable_coordinated_writes = enable_coordinated_writes
        self.coordinator_options = coordinator_options

        # Selection Feedback Loop
        self.selection_feedback_loop = selection_feedback_loop
        self.enable_performance_feedback = enable_performance_feedback

        # Post-debate workflow automation
        self.post_debate_workflow = post_debate_workflow
        self.enable_post_debate_workflow = enable_post_debate_workflow
        self.post_debate_workflow_threshold = post_debate_workflow_threshold

        # Callbacks
        self._emit_moment_event = emit_moment_event
        self._store_debate_outcome_as_memory = store_debate_outcome_as_memory
        self._update_continuum_memory_outcomes = update_continuum_memory_outcomes
        self._index_debate_async = index_debate_async
        self._ingest_debate_outcome = ingest_debate_outcome

        # Initialize helper classes for extracted logic
        self._consensus_storage = ConsensusStorage(consensus_memory=consensus_memory)
        self._training_emitter = TrainingEmitter(
            training_exporter=training_exporter,
            event_emitter=event_emitter,
            insight_store=insight_store,
            loop_id=loop_id,
        )

    async def execute(self, ctx: "DebateContext") -> None:
        """
        Execute all feedback loops.

        Args:
            ctx: The DebateContext with completed debate
        """
        if not ctx.result:
            logger.warning("FeedbackPhase called without result")
            return

        # 1. Record ELO match results
        self._record_elo_match(ctx)

        # 2. Update PersonaManager
        self._update_persona_performance(ctx)

        # 3. Resolve positions in PositionLedger
        self._resolve_positions(ctx)

        # 4. Update relationship metrics
        self._update_relationships(ctx)

        # 5. Detect narrative moments
        self._detect_moments(ctx)

        # 6. Index debate in embeddings
        await self._index_debate(ctx)

        # 7. Detect position flips
        self._detect_flips(ctx)

        # 8. Store debate outcome in ConsensusMemory for historical retrieval
        consensus_id = self._consensus_storage.store_consensus_outcome(ctx)
        if consensus_id:
            setattr(ctx, "_last_consensus_id", consensus_id)

        # 9. Store belief cruxes for future seeding
        self._consensus_storage.store_cruxes(ctx, consensus_id)

        # 10. Store debate outcome in ContinuumMemory
        self._store_memory(ctx)

        # 11. Update memory outcomes
        self._update_memory_outcomes(ctx)

        # 12. Record calibration data for prediction accuracy
        self._record_calibration(ctx)

        # 13. Update genome fitness for Genesis evolution
        self._update_genome_fitness(ctx)

        # 14. Maybe trigger population evolution
        await self._maybe_evolve_population(ctx)

        # 15. Record pulse outcome if debate was on a trending topic
        self._record_pulse_outcome(ctx)

        # 16. Run periodic memory cleanup
        self._run_memory_cleanup(ctx)

        # 17. Record evolution patterns from high-confidence debates
        self._record_evolution_patterns(ctx)

        # 18. Assess domain risks and emit warnings
        self._assess_risks(ctx)

        # 19. Record insight usage for learning loop (B2)
        await self._training_emitter.record_insight_usage(ctx)

        # 20. Emit training data for Tinker fine-tuning
        await self._training_emitter.emit_training_data(ctx)

        # 21. Ingest debate outcome into Knowledge Mound
        await self._ingest_knowledge_outcome(ctx)

        # 22. Store collected evidence in Knowledge Mound via EvidenceBridge
        await self._store_evidence_in_mound(ctx)

        # 23. Observe debate for culture patterns
        await self._observe_debate_culture(ctx)

        # 24. Auto-trigger broadcast for high-quality debates
        await self._maybe_trigger_broadcast(ctx)

        # 25. Execute coordinated memory writes (alternative to individual writes)
        await self._execute_coordinated_writes(ctx)

        # 26. Update selection feedback loop with debate outcome
        await self._update_selection_feedback(ctx)

        # 27. Trigger post-debate workflow for high-quality debates
        await self._maybe_trigger_workflow(ctx)

    async def _maybe_trigger_workflow(self, ctx: "DebateContext") -> None:
        """Trigger post-debate workflow for high-quality debates.

        When enabled via enable_post_debate_workflow, this method:
        1. Checks if debate confidence meets workflow threshold
        2. Triggers the workflow engine asynchronously (fire-and-forget)
        3. Logs success/failure without blocking debate completion

        Workflows can be used for automated refinement, documentation,
        knowledge extraction, or other post-debate processing.
        """
        if not self.post_debate_workflow or not self.enable_post_debate_workflow:
            return

        result = ctx.result
        if not result:
            return

        # Check confidence threshold
        confidence = getattr(result, "confidence", 0.0)
        if confidence < self.post_debate_workflow_threshold:
            logger.debug(
                "[workflow] Skipping workflow: confidence %.2f < threshold %.2f",
                confidence,
                self.post_debate_workflow_threshold,
            )
            return

        # Fire-and-forget workflow to not block debate completion
        asyncio.create_task(self._run_workflow_async(ctx))
        logger.info(
            "[workflow] Triggered post-debate workflow for debate %s (confidence=%.2f)",
            ctx.debate_id,
            confidence,
        )

    async def _run_workflow_async(self, ctx: "DebateContext") -> None:
        """Run workflow engine asynchronously.

        This is a fire-and-forget task so it doesn't block debate completion.
        Failures are logged but don't affect the debate result.
        """
        try:
            from aragora.workflow.engine import WorkflowEngine

            engine = WorkflowEngine()
            workflow_input = {
                "debate_id": ctx.debate_id,
                "task": ctx.env.task,
                "result": {
                    "winner": ctx.result.winner if ctx.result else None,
                    "confidence": ctx.result.confidence if ctx.result else 0.0,
                    "consensus_reached": ctx.result.consensus_reached if ctx.result else False,
                    "final_answer": ctx.result.final_answer if ctx.result else "",
                },
                "domain": ctx.domain,
                "agents": [a.name for a in ctx.agents] if ctx.agents else [],
            }

            workflow_result = await engine.execute(
                self.post_debate_workflow,
                input_data=workflow_input,
            )

            if workflow_result.success:
                logger.info(
                    "[workflow] Completed workflow for %s: output=%s",
                    ctx.debate_id,
                    str(workflow_result.output)[:200] if workflow_result.output else "None",
                )
            else:
                logger.warning(
                    "[workflow] Workflow failed for %s: %s",
                    ctx.debate_id,
                    workflow_result.error_message,
                )

        except ImportError:
            logger.debug("[workflow] WorkflowEngine not available")
        except Exception as e:
            logger.warning("[workflow] Post-debate workflow failed: %s", e)

    async def _execute_coordinated_writes(self, ctx: "DebateContext") -> None:
        """Execute coordinated atomic writes to all memory systems.

        When enabled, this provides transaction semantics for multi-system
        writes with rollback on partial failure. This is an alternative to
        the individual write operations in steps 8-11.

        Note: Individual writes still run for backward compatibility.
        The coordinator can be used for additional atomic operations.
        """
        if not self.memory_coordinator or not self.enable_coordinated_writes:
            return

        result = ctx.result
        if not result:
            return

        try:
            transaction = await self.memory_coordinator.commit_debate_outcome(
                ctx=ctx,
                options=self.coordinator_options,
            )

            if transaction.success:
                logger.info(
                    "[coordinator] Committed %d writes for debate %s",
                    len(transaction.operations),
                    ctx.debate_id,
                )
            elif transaction.partial_failure:
                failed = transaction.get_failed_operations()
                logger.warning(
                    "[coordinator] Partial failure for debate %s: %d/%d failed",
                    ctx.debate_id,
                    len(failed),
                    len(transaction.operations),
                )
                for op in failed:
                    logger.warning("[coordinator] Failed: %s - %s", op.target, op.error)

            # Store transaction reference in context for debugging
            setattr(ctx, "_memory_transaction", transaction)

        except Exception as e:
            logger.error("[coordinator] Transaction failed for %s: %s", ctx.debate_id, e)

    async def _update_selection_feedback(self, ctx: "DebateContext") -> None:
        """Update selection feedback loop with debate outcome.

        Records debate performance metrics and computes selection weight
        adjustments for participating agents based on their performance.
        """
        if not self.selection_feedback_loop or not self.enable_performance_feedback:
            return

        result = ctx.result
        if not result:
            return

        try:
            adjustments = self.selection_feedback_loop.process_debate_outcome(
                debate_id=ctx.debate_id,
                participants=[a.name for a in ctx.agents],
                winner=result.winner,
                domain=ctx.domain,
            )

            if adjustments:
                logger.debug(
                    "[feedback] Updated selection weights for %d agents",
                    len(adjustments),
                )

                # Emit selection feedback event
                self._emit_selection_feedback_event(ctx, adjustments)

        except Exception as e:
            logger.debug("[feedback] Selection feedback update failed: %s", e)

    def _emit_selection_feedback_event(
        self, ctx: "DebateContext", adjustments: Dict[str, float]
    ) -> None:
        """Emit SELECTION_FEEDBACK event for real-time monitoring."""
        if not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            self.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.SELECTION_FEEDBACK,
                    loop_id=self.loop_id,
                    data={
                        "debate_id": ctx.debate_id,
                        "adjustments": adjustments,
                        "domain": ctx.domain,
                        "winner": ctx.result.winner if ctx.result else None,
                    },
                )
            )
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.debug("Selection feedback event emission error: %s", e)

    async def _ingest_knowledge_outcome(self, ctx: "DebateContext") -> None:
        """Ingest debate outcome into Knowledge Mound for future retrieval.

        Stores high-confidence debate conclusions in the unified knowledge
        superstructure so they can inform future debates on related topics.
        """
        if not self.knowledge_mound or not self.enable_knowledge_ingestion:
            return

        if not self._ingest_debate_outcome:
            return

        result = ctx.result
        if not result:
            return

        try:
            await self._ingest_debate_outcome(result)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.warning("[knowledge_mound] Failed to ingest outcome: %s", e)

    async def _store_evidence_in_mound(self, ctx: "DebateContext") -> None:
        """Store collected evidence in Knowledge Mound via EvidenceBridge.

        Persists evidence gathered during the debate (sources, quotes, data)
        in the Knowledge Mound so it can inform future debates on related topics.
        """
        if not self.knowledge_bridge_hub:
            return

        evidence_items = getattr(ctx, "collected_evidence", [])
        if not evidence_items:
            return

        try:
            evidence_bridge = self.knowledge_bridge_hub.evidence
            stored_count = 0

            for evidence in evidence_items:
                try:
                    await evidence_bridge.store_from_collector_evidence(evidence)
                    stored_count += 1
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug("[evidence] Failed to store single item: %s", e)

            if stored_count > 0:
                logger.info(
                    "[evidence] Stored %d/%d evidence items in mound for debate %s",
                    stored_count,
                    len(evidence_items),
                    ctx.debate_id,
                )

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.warning("[evidence] Failed to store evidence: %s", e)

    async def _observe_debate_culture(self, ctx: "DebateContext") -> None:
        """Observe debate for organizational culture patterns.

        Extracts patterns from debate outcomes to build institutional knowledge
        about effective debate strategies, common arguments, and consensus patterns.
        This feeds into the CultureAccumulator for organizational learning.
        """
        if not self.knowledge_mound or not ctx.result:
            return

        try:
            # Check if the knowledge mound has observe_debate method
            if not hasattr(self.knowledge_mound, "observe_debate"):
                return

            patterns = await self.knowledge_mound.observe_debate(ctx.result)

            if patterns:
                logger.info(
                    "[culture] Extracted %d patterns from debate %s",
                    len(patterns),
                    ctx.debate_id,
                )

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.debug("[culture] Pattern observation failed: %s", e)

    async def _maybe_trigger_broadcast(self, ctx: "DebateContext") -> None:
        """Trigger broadcast for high-quality debates.

        When enabled via auto_broadcast, this method:
        1. Checks if debate confidence meets broadcast threshold
        2. Triggers the broadcast pipeline asynchronously (fire-and-forget)
        3. Logs success/failure without blocking debate completion

        This enables automatic podcast/video generation for noteworthy debates.
        """
        if not self.broadcast_pipeline or not self.auto_broadcast:
            return

        result = ctx.result
        if not result:
            return

        # Check confidence threshold
        confidence = getattr(result, "confidence", 0.0)
        if confidence < self.broadcast_min_confidence:
            logger.debug(
                "[broadcast] Skipping broadcast: confidence %.2f < threshold %.2f",
                confidence,
                self.broadcast_min_confidence,
            )
            return

        # Fire-and-forget broadcast to not block debate completion
        asyncio.create_task(self._broadcast_async(ctx))
        logger.info(
            "[broadcast] Triggered auto-broadcast for debate %s (confidence=%.2f)",
            ctx.debate_id,
            confidence,
        )

    async def _broadcast_async(self, ctx: "DebateContext") -> None:
        """Run broadcast pipeline asynchronously.

        This is a fire-and-forget task so it doesn't block debate completion.
        Failures are logged but don't affect the debate result.
        """
        try:
            from aragora.broadcast.pipeline import BroadcastOptions

            options = BroadcastOptions(
                audio_enabled=True,
                generate_rss_episode=True,
            )

            pipeline_result = await self.broadcast_pipeline.run(
                ctx.debate_id,
                options,
            )

            if pipeline_result.success:
                logger.info(
                    "[broadcast] Successfully generated broadcast for %s: audio=%s",
                    ctx.debate_id,
                    pipeline_result.audio_path,
                )
            else:
                logger.warning(
                    "[broadcast] Broadcast failed for %s: %s",
                    ctx.debate_id,
                    pipeline_result.error_message,
                )

        except Exception as e:
            logger.warning("[broadcast] Auto-broadcast failed: %s", e)

    def _assess_risks(self, ctx: "DebateContext") -> None:
        """Assess domain-specific risks and emit RISK_WARNING events.

        Analyzes the debate topic for safety-sensitive domains (medical, legal,
        financial, etc.) and emits warnings for real-time panel updates.
        """
        if not self.event_emitter:
            return

        try:
            from aragora.debate.risk_assessor import assess_debate_risk
            from aragora.server.stream import StreamEvent, StreamEventType

            # Assess risks for the debate topic
            risks = assess_debate_risk(ctx.env.task, domain=ctx.domain)

            for risk in risks:
                self.event_emitter.emit(
                    StreamEvent(
                        type=StreamEventType.RISK_WARNING,
                        loop_id=self.loop_id,
                        data={
                            "level": risk.level.value,
                            "domain": risk.domain,
                            "category": risk.category,
                            "description": risk.description,
                            "mitigations": risk.mitigations,
                            "confidence": risk.confidence,
                            "debate_id": ctx.debate_id,
                        },
                    )
                )

            if risks:
                logger.info(
                    "[risk] Identified %d risks for debate %s: %s",
                    len(risks),
                    ctx.debate_id,
                    [r.level.value for r in risks],
                )

        except ImportError:
            logger.debug("Risk assessment unavailable: module not found")
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.debug(f"Risk assessment error: {e}")

    def _record_calibration(self, ctx: "DebateContext") -> None:
        """Record calibration data from agent votes with confidence.

        Tracks (confidence, outcome) pairs for measuring prediction accuracy.
        Each vote with confidence is recorded as a calibration data point,
        comparing the voted choice against the actual debate winner.
        """
        if not self.calibration_tracker:
            return

        result = ctx.result
        if not result:
            return

        # Determine the actual outcome
        actual_winner = result.winner or "no_consensus"

        try:
            # Record each vote with confidence as a calibration data point
            recorded = 0
            for vote in result.votes:
                # Check if vote has confidence attribute
                confidence = getattr(vote, "confidence", None)
                if confidence is None:
                    continue

                # Determine if the prediction was correct
                # A vote is "correct" if the voted choice matches the actual winner
                correct = vote.choice == actual_winner

                # Record the prediction with correct parameter names
                self.calibration_tracker.record_prediction(
                    agent=vote.agent,
                    confidence=confidence,
                    correct=correct,
                    debate_id=getattr(result, "debate_id", ""),
                )
                recorded += 1

            if recorded > 0:
                logger.debug(f"[calibration] Recorded {recorded} predictions")
                # Emit CALIBRATION_UPDATE event for real-time panel updates
                self._emit_calibration_update(ctx, recorded)
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"[calibration] Failed to record: {e}")

    def _emit_calibration_update(self, ctx: "DebateContext", recorded_count: int) -> None:
        """Emit CALIBRATION_UPDATE event to WebSocket."""
        if not self.event_emitter or not self.calibration_tracker:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            # Get summary stats from calibration tracker
            summary = {}
            if hasattr(self.calibration_tracker, "get_summary"):
                summary = self.calibration_tracker.get_summary()

            self.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.CALIBRATION_UPDATE,
                    loop_id=self.loop_id,
                    data={
                        "debate_id": ctx.debate_id,
                        "predictions_recorded": recorded_count,
                        "total_predictions": summary.get("total_predictions", 0),
                        "overall_accuracy": summary.get("overall_accuracy", 0.0),
                        "domain": ctx.domain,
                    },
                )
            )
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.debug(f"Calibration event emission error: {e}")

    def _record_pulse_outcome(self, ctx: "DebateContext") -> None:
        """Record pulse outcome if the debate was on a trending topic.

        This enables analytics on which trending topics lead to productive debates.
        """
        if not self.pulse_manager:
            return

        result = ctx.result
        if not result:
            return

        # Check if the debate has a trending topic attached
        trending_topic = getattr(ctx, "trending_topic", None)
        if not trending_topic:
            # Also check arena for backwards compatibility
            arena = getattr(ctx, "arena", None)
            if arena:
                trending_topic = getattr(arena, "trending_topic", None)

        if not trending_topic:
            return

        try:
            self.pulse_manager.record_debate_outcome(
                topic=getattr(trending_topic, "topic", str(trending_topic)),
                platform=getattr(trending_topic, "platform", "unknown"),
                debate_id=ctx.debate_id,
                consensus_reached=result.consensus_reached,
                confidence=result.confidence,
                rounds_used=result.rounds_used,
                category=getattr(trending_topic, "category", ""),
                volume=getattr(trending_topic, "volume", 0),
            )
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.warning(f"[pulse] Failed to record outcome: {e}")

    def _run_memory_cleanup(self, ctx: "DebateContext") -> None:
        """Run periodic memory cleanup to prevent unbounded growth.

        Cleans up expired memories and enforces tier limits. This activates
        the previously stranded cleanup functionality in ContinuumMemory.

        Cleanup runs:
        - cleanup_expired_memories(): Every debate
        - enforce_tier_limits(): 10% of debates (probabilistic)
        """
        if not self.continuum_memory:
            return

        import random

        try:
            # Always try to clean expired memories
            cleaned = self.continuum_memory.cleanup_expired_memories()
            if cleaned > 0:
                logger.debug(f"[memory] Cleaned {cleaned} expired memories")

            # Probabilistically enforce tier limits (10% of debates)
            if random.random() < 0.1:
                self.continuum_memory.enforce_tier_limits()
                logger.debug("[memory] Enforced tier limits")

        except (TypeError, ValueError, AttributeError, OSError, RuntimeError) as e:
            logger.debug(f"[memory] Cleanup error (non-fatal): {e}")

    def _record_elo_match(self, ctx: "DebateContext") -> None:
        """Record ELO match results."""
        if not self.elo_system:
            return

        result = ctx.result
        if not result.winner:
            return

        try:
            participants = [agent.name for agent in ctx.agents]
            scores = {}

            for agent_name in participants:
                if agent_name == result.winner:
                    scores[agent_name] = 1.0
                elif result.consensus_reached:
                    scores[agent_name] = 0.5  # Draw for non-winners in consensus
                else:
                    scores[agent_name] = 0.0

            self.elo_system.record_match(ctx.debate_id, participants, scores, domain=ctx.domain)

            # Emit MATCH_RECORDED event
            self._emit_match_recorded_event(ctx, participants)

        except Exception as e:
            _, msg, exc_info = _build_error_action(e, "elo")
            logger.warning(
                "ELO update failed for debate %s: %s", ctx.debate_id, msg, exc_info=exc_info
            )

    def _emit_match_recorded_event(self, ctx: "DebateContext", participants: list[str]) -> None:
        """Emit MATCH_RECORDED event for real-time leaderboard updates."""
        if not self.event_emitter or not self.elo_system:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            # Batch fetch all ratings
            ratings_batch = self.elo_system.get_ratings_batch(participants)
            elo_changes = {
                name: ratings_batch[name].elo if name in ratings_batch else 1500.0
                for name in participants
            }

            self.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.MATCH_RECORDED,
                    loop_id=self.loop_id,
                    data={
                        "debate_id": ctx.debate_id,
                        "participants": participants,
                        "elo_changes": elo_changes,
                        "domain": ctx.domain,
                        "winner": ctx.result.winner,
                    },
                )
            )

            # Emit per-agent ELO updates for granular tracking
            for agent_name in participants:
                rating = ratings_batch.get(agent_name)
                if rating:
                    self.event_emitter.emit(
                        StreamEvent(
                            type=StreamEventType.AGENT_ELO_UPDATED,
                            loop_id=self.loop_id,
                            agent=agent_name,
                            data={
                                "agent": agent_name,
                                "new_elo": rating.elo,
                                "debate_id": ctx.debate_id,
                                "domain": ctx.domain,
                                "is_winner": agent_name == ctx.result.winner,
                            },
                        )
                    )
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.warning(f"ELO event emission error: {e}")

    def _update_persona_performance(self, ctx: "DebateContext") -> None:
        """Update PersonaManager with performance feedback."""
        if not self.persona_manager:
            return

        try:
            result = ctx.result
            for agent in ctx.agents:
                success = (agent.name == result.winner) or (
                    result.consensus_reached and result.confidence > 0.7
                )
                self.persona_manager.record_performance(
                    agent_name=agent.name,
                    domain=ctx.domain,
                    success=success,
                )

            # Check for trait emergence after performance updates
            self._check_trait_emergence(ctx)
        except Exception as e:
            _, msg, exc_info = _build_error_action(e, "persona")
            logger.warning("Persona update failed: %s", msg, exc_info=exc_info)

    def _check_trait_emergence(self, ctx: "DebateContext") -> None:
        """Check if any new agent traits emerged from performance patterns.

        Traits emerge when an agent demonstrates consistent behavior patterns:
        - High win rates in specific domains
        - Consistent prediction accuracy
        - Distinct communication styles
        """
        if not self.persona_manager or not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            for agent in ctx.agents:
                # Get agent's current traits
                persona = self.persona_manager.get_persona(agent.name)
                if not persona:
                    continue

                # Check for newly emerged traits
                new_traits = getattr(persona, "emerging_traits", [])
                if not new_traits:
                    # Try to detect traits from performance history
                    new_traits = self._detect_emerging_traits(agent.name, ctx)

                for trait in new_traits:
                    self.event_emitter.emit(
                        StreamEvent(
                            type=StreamEventType.TRAIT_EMERGED,
                            loop_id=self.loop_id,
                            data={
                                "agent": agent.name,
                                "trait": trait.get("name", "unknown"),
                                "description": trait.get("description", ""),
                                "confidence": trait.get("confidence", 0.5),
                                "domain": ctx.domain,
                                "debate_id": ctx.debate_id,
                            },
                        )
                    )
                    logger.info(
                        "[persona] Trait emerged for %s: %s",
                        agent.name,
                        trait.get("name", "unknown"),
                    )

        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.debug(f"Trait emergence check error: {e}")

    def _detect_emerging_traits(
        self, agent_name: str, ctx: "DebateContext"
    ) -> list[dict[str, Any]]:
        """Detect traits based on agent performance patterns.

        Returns list of trait dicts with name, description, confidence.
        """
        traits: list[dict[str, Any]] = []

        try:
            # Get performance stats if available
            if not hasattr(self.persona_manager, "get_performance_stats"):
                return traits

            stats = self.persona_manager.get_performance_stats(agent_name)
            if not stats:
                return traits

            # Domain specialist: High win rate in specific domain
            domain_wins = stats.get("domain_wins", {})
            if ctx.domain in domain_wins and domain_wins[ctx.domain] >= 3:
                traits.append(
                    {
                        "name": f"{ctx.domain}_specialist",
                        "description": f"Demonstrated expertise in {ctx.domain} domain",
                        "confidence": min(0.9, 0.5 + (domain_wins[ctx.domain] * 0.1)),
                    }
                )

            # High calibration: Consistent accurate predictions
            accuracy = stats.get("prediction_accuracy", 0.0)
            if accuracy >= 0.8 and stats.get("total_predictions", 0) >= 5:
                traits.append(
                    {
                        "name": "well_calibrated",
                        "description": f"Highly accurate predictions ({accuracy:.0%})",
                        "confidence": accuracy,
                    }
                )

            # Consistent winner: High overall win rate
            win_rate = stats.get("win_rate", 0.0)
            if win_rate >= 0.7 and stats.get("total_debates", 0) >= 5:
                traits.append(
                    {
                        "name": "consistent_winner",
                        "description": f"Wins {win_rate:.0%} of debates",
                        "confidence": win_rate,
                    }
                )

        except (TypeError, ValueError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.debug(f"Trait detection error for {agent_name}: {e}")

        return traits

    def _resolve_positions(self, ctx: "DebateContext") -> None:
        """Resolve positions in PositionLedger."""
        if not self.position_ledger:
            return

        result = ctx.result
        if not result.final_answer:
            return

        try:
            for agent in ctx.agents:
                positions = self.position_ledger.get_agent_positions(agent.name)
                for pos in positions[-5:]:  # Last 5 positions
                    if pos.debate_id == ctx.debate_id:
                        outcome = "correct" if agent.name == result.winner else "contested"
                        self.position_ledger.resolve_position(
                            position_id=pos.id,
                            outcome=outcome,
                            resolution_source=f"debate:{ctx.debate_id}",
                        )
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("Position resolution failed: %s", e)

    def _update_relationships(self, ctx: "DebateContext") -> None:
        """Update relationship metrics from debate."""
        if not self.relationship_tracker:
            return

        try:
            result = ctx.result

            # Extract critiques from messages
            critiques = []
            if result.messages:
                for msg in result.messages:
                    if getattr(msg, "role", "") == "critic":
                        critiques.append(
                            {
                                "agent": getattr(msg, "agent", "unknown"),
                                "target": getattr(msg, "target_agent", None),
                            }
                        )

            # Build vote mapping
            votes = {}
            for v in result.votes:
                canonical = ctx.choice_mapping.get(v.choice, v.choice)
                votes[v.agent] = canonical

            self.relationship_tracker.update_from_debate(
                debate_id=ctx.debate_id,
                participants=[agent.name for agent in ctx.agents],
                winner=result.winner,
                votes=votes,
                critiques=critiques,
            )
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("Relationship tracking failed: %s", e)

    def _detect_moments(self, ctx: "DebateContext") -> None:
        """Detect significant narrative moments."""
        if not self.moment_detector:
            return

        try:
            result = ctx.result

            # Upset victories
            if result.winner and self.elo_system:
                for agent in ctx.agents:
                    if agent.name != result.winner:
                        moment = self.moment_detector.detect_upset_victory(
                            winner=result.winner,
                            loser=agent.name,
                            debate_id=ctx.debate_id,
                        )
                        if moment:
                            self.moment_detector.record_moment(moment)
                            if self._emit_moment_event:
                                self._emit_moment_event(moment)

            # Calibration vindications
            for v in result.votes:
                if v.confidence >= 0.85:
                    canonical = ctx.choice_mapping.get(v.choice, v.choice)
                    was_correct = canonical == result.winner
                    if was_correct:
                        moment = self.moment_detector.detect_calibration_vindication(
                            agent_name=v.agent,
                            prediction_confidence=v.confidence,
                            was_correct=True,
                            domain=ctx.domain,
                            debate_id=ctx.debate_id,
                        )
                        if moment:
                            self.moment_detector.record_moment(moment)
                            if self._emit_moment_event:
                                self._emit_moment_event(moment)

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("Moment detection failed: %s", e)

    async def _index_debate(self, ctx: "DebateContext") -> None:
        """Index debate in embeddings for historical retrieval."""
        if not self.debate_embeddings:
            return

        try:
            result = ctx.result

            # Build transcript
            transcript_parts = []
            if result.messages:
                for msg in result.messages[:30]:
                    agent_name = getattr(msg, "agent", "unknown")
                    content = getattr(msg, "content", str(msg))[:500]
                    transcript_parts.append(f"{agent_name}: {content}")

            artifact = {
                "id": ctx.debate_id,
                "task": ctx.env.task,
                "domain": ctx.domain,
                "winner": result.winner,
                "final_answer": result.final_answer or "",
                "confidence": result.confidence,
                "agents": [a.name for a in ctx.agents],
                "transcript": "\n".join(transcript_parts),
                "rounds_used": result.rounds_used,
                "consensus_reached": result.consensus_reached,
            }

            if self._index_debate_async:
                task = asyncio.create_task(self._index_debate_async(artifact))
                task.add_done_callback(
                    lambda t: (
                        logger.warning(f"Debate indexing failed: {t.exception()}")
                        if t.exception()
                        else None
                    )
                )

        except (
            TypeError,
            ValueError,
            AttributeError,
            KeyError,
            RuntimeError,
            OSError,
            ConnectionError,
        ) as e:
            logger.warning("Embedding indexing failed: %s", e)

    def _detect_flips(self, ctx: "DebateContext") -> None:
        """Detect position flips for all participating agents."""
        if not self.flip_detector:
            return

        try:
            for agent in ctx.agents:
                flips = self.flip_detector.detect_flips_for_agent(agent.name)
                if flips:
                    logger.info(
                        "[flip] Detected %d position changes for %s", len(flips), agent.name
                    )
                    self._emit_flip_events(ctx, agent.name, flips)

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("Flip detection failed: %s", e)

    def _emit_flip_events(self, ctx: "DebateContext", agent_name: str, flips: list) -> None:
        """Emit FLIP_DETECTED events to WebSocket."""
        if not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            for flip in flips:
                self.event_emitter.emit(
                    StreamEvent(
                        type=StreamEventType.FLIP_DETECTED,
                        loop_id=self.loop_id,
                        data={
                            "agent": agent_name,
                            "flip_type": getattr(flip, "flip_type", "unknown"),
                            "original_claim": getattr(flip, "original_claim", "")[:200],
                            "new_claim": getattr(flip, "new_claim", "")[:200],
                            "original_confidence": getattr(flip, "original_confidence", 0.0),
                            "new_confidence": getattr(flip, "new_confidence", 0.0),
                            "similarity_score": getattr(flip, "similarity_score", 0.0),
                            "domain": getattr(flip, "domain", None),
                            "debate_id": ctx.result.id if ctx.result else ctx.debate_id,
                        },
                    )
                )
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.warning(f"Flip event emission error: {e}")

    def _store_memory(self, ctx: "DebateContext") -> None:
        """Store debate outcome in ContinuumMemory."""
        if not self.continuum_memory:
            return

        result = ctx.result
        if not result.final_answer:
            return

        if self._store_debate_outcome_as_memory:
            self._store_debate_outcome_as_memory(result)

    def _update_memory_outcomes(self, ctx: "DebateContext") -> None:
        """Update retrieved memories based on debate outcome."""
        if not self.continuum_memory:
            return

        if self._update_continuum_memory_outcomes:
            self._update_continuum_memory_outcomes(ctx.result)

    def _update_genome_fitness(self, ctx: "DebateContext") -> None:
        """Update genome fitness scores based on debate outcome.

        For agents with genome_id attributes (evolved via Genesis),
        update their fitness scores based on debate performance.
        """
        if not self.population_manager:
            return

        result = ctx.result
        if not result:
            return

        winner_agent = getattr(result, "winner", None)

        for agent in ctx.agents:
            genome_id = getattr(agent, "genome_id", None)
            if not genome_id:
                continue

            try:
                # Determine if this agent won
                consensus_win = agent.name == winner_agent

                # Check if agent's prediction was correct
                prediction_correct = self._check_agent_prediction(agent, ctx)

                # Update fitness in population manager
                self.population_manager.update_fitness(
                    genome_id,
                    consensus_win=consensus_win,
                    prediction_correct=prediction_correct,
                )

                logger.debug(
                    "[genesis] Updated fitness for genome %s: win=%s pred=%s",
                    genome_id[:8],
                    consensus_win,
                    prediction_correct,
                )
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
                logger.debug("Genome fitness update failed for %s: %s", agent.name, e)

    def _check_agent_prediction(
        self,
        agent: "Agent",
        ctx: "DebateContext",
    ) -> bool:
        """Check if an agent correctly predicted the debate outcome.

        Returns True if the agent's vote matched the final winner.
        """
        result = ctx.result
        if not result or not result.votes:
            return False

        winner = getattr(result, "winner", None)
        if not winner:
            return False

        for vote in result.votes:
            if vote.agent == agent.name:
                # Check if the agent's choice matches the winner
                canonical = ctx.choice_mapping.get(vote.choice, vote.choice)
                return canonical == winner

        return False

    async def _maybe_evolve_population(self, ctx: "DebateContext") -> None:
        """Trigger population evolution after high-quality debates.

        Evolution is triggered when:
        1. auto_evolve is True
        2. Debate confidence >= breeding_threshold
        3. Population has accumulated enough debate history
        """
        if not self.population_manager or not self.auto_evolve:
            return

        result = ctx.result
        if not result:
            return

        # Only evolve after high-confidence debates
        if result.confidence < self.breeding_threshold:
            return

        try:
            # Get the population for these agents
            agent_names = [a.name for a in ctx.agents]
            population = self.population_manager.get_or_create_population(agent_names)

            if not population:
                return

            # Track debate in population history
            history = getattr(population, "debate_history", []) or []
            history.append(ctx.debate_id)

            # Evolve every 5 debates
            if len(history) % 5 == 0:
                # Fire-and-forget evolution
                asyncio.create_task(self._evolve_async(population))
                logger.info(
                    "[genesis] Triggered evolution after %d debates (confidence=%.2f)",
                    len(history),
                    result.confidence,
                )

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.debug("Evolution check failed: %s", e)

    async def _evolve_async(self, population: Any) -> None:
        """Run population evolution asynchronously.

        This is a fire-and-forget task so it doesn't block debate completion.
        """
        try:
            evolved = self.population_manager.evolve_population(population)
            logger.info(
                "[genesis] Population evolved to generation %d with %d genomes",
                evolved.generation,
                len(evolved.genomes),
            )

            # Emit event if event_emitter available
            if self.event_emitter:
                from aragora.server.stream import StreamEvent, StreamEventType

                self.event_emitter.emit(
                    StreamEvent(
                        type=StreamEventType.GENESIS_EVOLUTION,
                        loop_id=self.loop_id,
                        data={
                            "generation": evolved.generation,
                            "genome_count": len(evolved.genomes),
                            "population_id": getattr(population, "id", ""),
                            "top_fitness": getattr(evolved, "top_fitness", 0.0),
                        },
                    )
                )

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("[genesis] Evolution failed: %s", e)

    def _record_evolution_patterns(self, ctx: "DebateContext") -> None:
        """Extract winning patterns from high-confidence debates for prompt evolution.

        When enabled via protocol.enable_evolution, this method:
        1. Extracts patterns from successful debates (high confidence)
        2. Stores patterns in the PromptEvolver database
        3. Updates performance metrics for agent prompts

        Only runs for debates with confidence >= 0.7 to ensure quality patterns.
        """
        if not self.prompt_evolver:
            return

        result = ctx.result
        if not result:
            return

        # Only extract patterns from high-confidence debates
        if result.confidence < 0.7:
            return

        try:
            # Build a minimal DebateResult-like object for the evolver
            # The evolver expects objects with specific attributes
            class DebateResultProxy:
                def __init__(self, ctx_result, ctx_obj):
                    self.id = ctx_obj.debate_id
                    self.consensus_reached = ctx_result.consensus_reached
                    self.confidence = ctx_result.confidence
                    self.final_answer = ctx_result.final_answer or ""
                    self.critiques = []

                    # Extract critiques from messages if available
                    if ctx_result.messages:
                        for msg in ctx_result.messages:
                            if getattr(msg, "role", "") == "critic":
                                # Create a critique-like object
                                class CritiqueProxy:
                                    def __init__(self, m):
                                        self.severity = getattr(m, "severity", 0.5)
                                        self.issues = getattr(m, "issues", [])
                                        self.suggestions = getattr(m, "suggestions", [])

                                self.critiques.append(CritiqueProxy(msg))

            proxy = DebateResultProxy(result, ctx)

            # Extract patterns from this debate
            patterns = self.prompt_evolver.extract_winning_patterns([proxy])
            if patterns:
                self.prompt_evolver.store_patterns(patterns)
                logger.info(
                    "[evolution] Extracted %d patterns from debate %s (confidence=%.2f)",
                    len(patterns),
                    ctx.debate_id,
                    result.confidence,
                )

            # Update performance for each agent's current prompt version
            for agent in ctx.agents:
                prompt_version = getattr(agent, "prompt_version", None)
                if prompt_version is not None:
                    self.prompt_evolver.update_performance(
                        agent_name=agent.name,
                        version=prompt_version,
                        debate_result=proxy,
                    )

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.debug("[evolution] Pattern extraction failed: %s", e)

    # =========================================================================
    # Backward-compatible delegate methods
    # These proxy to extracted helper classes while maintaining the same API
    # that existing tests and callers rely on.
    # =========================================================================

    def _store_consensus_outcome(self, ctx: "DebateContext") -> None:
        """Delegate to ConsensusStorage for backward compatibility."""
        consensus_id = self._consensus_storage.store_consensus_outcome(ctx)
        if consensus_id:
            setattr(ctx, "_last_consensus_id", consensus_id)

    def _confidence_to_strength(self, confidence: float) -> "ConsensusStrength":
        """Delegate to ConsensusStorage for backward compatibility."""
        return self._consensus_storage._confidence_to_strength(confidence)

    def _store_cruxes(self, ctx: "DebateContext") -> None:
        """Delegate to ConsensusStorage for backward compatibility."""
        self._consensus_storage.store_cruxes(ctx)

    async def _record_insight_usage(self, ctx: "DebateContext") -> None:
        """Delegate to TrainingEmitter for backward compatibility."""
        await self._training_emitter.record_insight_usage(ctx)

    async def _emit_training_data(self, ctx: "DebateContext") -> None:
        """Delegate to TrainingEmitter for backward compatibility."""
        await self._training_emitter.emit_training_data(ctx)

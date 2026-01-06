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
"""

import asyncio
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Agent, DebateResult
    from aragora.debate.context import DebateContext

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
        elo_system: Any = None,
        persona_manager: Any = None,
        position_ledger: Any = None,
        relationship_tracker: Any = None,
        moment_detector: Any = None,
        debate_embeddings: Any = None,
        flip_detector: Any = None,
        continuum_memory: Any = None,
        event_emitter: Any = None,
        loop_id: Optional[str] = None,
        # Callbacks for orchestrator methods
        emit_moment_event: Optional[Callable] = None,
        store_debate_outcome_as_memory: Optional[Callable] = None,
        update_continuum_memory_outcomes: Optional[Callable] = None,
        index_debate_async: Optional[Callable] = None,
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

        # Callbacks
        self._emit_moment_event = emit_moment_event
        self._store_debate_outcome_as_memory = store_debate_outcome_as_memory
        self._update_continuum_memory_outcomes = update_continuum_memory_outcomes
        self._index_debate_async = index_debate_async

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

        # 8. Store debate outcome in ContinuumMemory
        self._store_memory(ctx)

        # 9. Update memory outcomes
        self._update_memory_outcomes(ctx)

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

            self.elo_system.record_match(
                ctx.debate_id, participants, scores, domain=ctx.domain
            )

            # Emit MATCH_RECORDED event
            self._emit_match_recorded_event(ctx, participants)

        except Exception as e:
            logger.debug("ELO update failed: %s", e)

    def _emit_match_recorded_event(
        self, ctx: "DebateContext", participants: list[str]
    ) -> None:
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

            self.event_emitter.emit(StreamEvent(
                type=StreamEventType.MATCH_RECORDED,
                loop_id=self.loop_id,
                data={
                    "debate_id": ctx.debate_id,
                    "participants": participants,
                    "elo_changes": elo_changes,
                    "domain": ctx.domain,
                    "winner": ctx.result.winner,
                }
            ))
        except Exception as e:
            logger.debug(f"ELO event emission error: {e}")

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
        except Exception as e:
            logger.debug("Persona update failed: %s", e)

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
        except Exception as e:
            logger.debug("Position resolution failed: %s", e)

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
                    if getattr(msg, 'role', '') == 'critic':
                        critiques.append({
                            "agent": getattr(msg, 'agent', 'unknown'),
                            "target": getattr(msg, 'target_agent', None),
                        })

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
        except Exception as e:
            logger.debug("Relationship tracking failed: %s", e)

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
                    was_correct = (canonical == result.winner)
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

        except Exception as e:
            logger.debug("Moment detection failed: %s", e)

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
                    agent_name = getattr(msg, 'agent', 'unknown')
                    content = getattr(msg, 'content', str(msg))[:500]
                    transcript_parts.append(f"{agent_name}: {content}")

            artifact = {
                'id': ctx.debate_id,
                'task': ctx.env.task,
                'domain': ctx.domain,
                'winner': result.winner,
                'final_answer': result.final_answer or '',
                'confidence': result.confidence,
                'agents': [a.name for a in ctx.agents],
                'transcript': '\n'.join(transcript_parts),
                'rounds_used': result.rounds_used,
                'consensus_reached': result.consensus_reached,
            }

            if self._index_debate_async:
                task = asyncio.create_task(self._index_debate_async(artifact))
                task.add_done_callback(lambda t: (
                    logger.warning(f"Debate indexing failed: {t.exception()}")
                    if t.exception() else None
                ))

        except Exception as e:
            logger.debug("Embedding indexing failed: %s", e)

    def _detect_flips(self, ctx: "DebateContext") -> None:
        """Detect position flips for all participating agents."""
        if not self.flip_detector:
            return

        try:
            for agent in ctx.agents:
                flips = self.flip_detector.detect_flips_for_agent(agent.name)
                if flips:
                    logger.info(
                        "[flip] Detected %d position changes for %s",
                        len(flips), agent.name
                    )
                    self._emit_flip_events(ctx, agent.name, flips)

        except Exception as e:
            logger.debug("Flip detection failed: %s", e)

    def _emit_flip_events(
        self, ctx: "DebateContext", agent_name: str, flips: list
    ) -> None:
        """Emit FLIP_DETECTED events to WebSocket."""
        if not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            for flip in flips:
                self.event_emitter.emit(StreamEvent(
                    type=StreamEventType.FLIP_DETECTED,
                    loop_id=self.loop_id,
                    data={
                        "agent": agent_name,
                        "flip_type": getattr(flip, 'flip_type', 'unknown'),
                        "original_claim": getattr(flip, 'original_claim', '')[:200],
                        "new_claim": getattr(flip, 'new_claim', '')[:200],
                        "original_confidence": getattr(flip, 'original_confidence', 0.0),
                        "new_confidence": getattr(flip, 'new_confidence', 0.0),
                        "similarity_score": getattr(flip, 'similarity_score', 0.0),
                        "domain": getattr(flip, 'domain', None),
                        "debate_id": ctx.result.id if ctx.result else ctx.debate_id,
                    }
                ))
        except Exception as e:
            logger.debug(f"Flip event emission error: {e}")

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

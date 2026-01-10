"""
PostDebateProcessor: Centralized post-debate processing for the nomic loop.

Handles all 12+ sequential processing steps that occur after a debate completes:
1. Consensus storage
2. Calibration recording
3. Suggestion feedback
4. Insight extraction
5. Agent memory recording
6. Persona performance
7. Pattern extraction
8. Meta-critique analysis
9. ELO rating updates
10. Domain calibration
11. Position ledger updates
12. Relationship tracking

This module is part of Wave 3 extraction from nomic_loop.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult


@dataclass
class ProcessingDependencies:
    """Container for all post-debate processing dependencies.

    Groups dependencies by purpose. All are optional - the processor
    gracefully skips steps when dependencies are unavailable.
    """
    # Memory systems
    consensus_memory: Optional[Any] = None
    insight_store: Optional[Any] = None
    memory_stream: Optional[Any] = None
    pattern_store: Optional[Any] = None
    critique_store: Optional[Any] = None

    # Tracking systems
    calibration_tracker: Optional[Any] = None
    suggestion_feedback: Optional[Any] = None
    elo_system: Optional[Any] = None
    position_ledger: Optional[Any] = None
    relationship_tracker: Optional[Any] = None

    # Agent systems
    persona_manager: Optional[Any] = None
    prompt_evolver: Optional[Any] = None

    # Feature availability flags
    consensus_memory_available: bool = False
    calibration_available: bool = False
    insight_available: bool = False
    elo_available: bool = False
    grounded_available: bool = False


@dataclass
class ProcessingContext:
    """Context for a single post-debate processing run.

    Captures all the information needed to process a debate result.
    """
    result: Any  # DebateResult
    agents: List[Any]
    topic: str = ""
    domain: str = "general"
    cycle: int = 0
    debate_id: str = ""


class PostDebateProcessor:
    """Executes all post-debate processing in sequence.

    Centralizes the 12+ processing steps that occur after each debate:
    - Stores consensus for future retrieval
    - Records calibration data for accuracy tracking
    - Extracts insights for learning
    - Updates agent memories and relationships
    - Adjusts ELO ratings

    All steps are fault-tolerant - failures in one step don't block others.

    Usage:
        processor = PostDebateProcessor(deps, log_fn=self._log)
        await processor.process(ProcessingContext(
            result=debate_result,
            agents=debate_team,
            topic="Improve error handling",
            domain="engineering",
            cycle=42,
        ))
    """

    def __init__(
        self,
        deps: ProcessingDependencies,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit: Optional[Callable[[str, dict], None]] = None,
        detect_domain_fn: Optional[Callable[[str], str]] = None,
        agent_in_consensus_fn: Optional[Callable[[str, Any], bool]] = None,
    ):
        """Initialize processor with dependencies.

        Args:
            deps: Container holding all processing dependencies
            log_fn: Optional logging callback
            stream_emit: Optional streaming callback for events
            detect_domain_fn: Optional domain detection callback
            agent_in_consensus_fn: Optional consensus checking callback
        """
        self.deps = deps
        self._log = log_fn or (lambda msg: print(f"[post-process] {msg}"))
        self._stream_emit = stream_emit
        self._detect_domain = detect_domain_fn or (lambda t: "general")
        self._agent_in_consensus = agent_in_consensus_fn or (lambda a, r: False)

    async def process(self, ctx: ProcessingContext) -> None:
        """Execute all post-debate processing steps.

        Steps are executed in order, with each step isolated for fault tolerance.
        A failure in one step logs the error but doesn't prevent other steps.

        Args:
            ctx: ProcessingContext with debate result and metadata
        """
        # Generate debate ID if not provided
        if not ctx.debate_id:
            ctx.debate_id = f"cycle-{ctx.cycle}-debate"

        # Detect domain if not provided
        if ctx.domain == "general" and ctx.topic:
            ctx.domain = self._detect_domain(ctx.topic)

        # Execute all processing steps
        await self._store_consensus(ctx)
        self._record_calibration(ctx)
        self._record_suggestion_feedback(ctx)
        await self._extract_insights(ctx)
        await self._record_agent_memories(ctx)
        self._record_persona_performance(ctx)
        await self._extract_patterns(ctx)
        self._analyze_meta_critique(ctx)
        self._record_elo_match(ctx)
        self._record_domain_calibration(ctx)
        self._record_positions(ctx)
        self._update_relationships(ctx)
        self._update_reputation(ctx)

    async def _store_consensus(self, ctx: ProcessingContext) -> None:
        """Store debate consensus for future reference (P1: ConsensusMemory)."""
        if not self.deps.consensus_memory or not self.deps.consensus_memory_available:
            return

        try:
            if ctx.result.consensus_reached and ctx.result.final_answer:
                await self.deps.consensus_memory.store(
                    topic=ctx.topic or "improvement debate",
                    consensus=ctx.result.final_answer,
                    participants=[a.name for a in ctx.agents],
                    confidence=getattr(ctx.result, 'confidence', 0.7),
                )
                self._log(f"  [consensus] Stored consensus for '{ctx.topic[:50]}...'")
        except Exception as e:
            self._log(f"  [consensus] Storage failed: {e}")

    def _record_calibration(self, ctx: ProcessingContext) -> None:
        """Record calibration data from debate predictions (P10: CalibrationTracker)."""
        if not self.deps.calibration_tracker or not self.deps.calibration_available:
            return

        try:
            for agent in ctx.agents:
                predicted_confidence = getattr(ctx.result, 'confidence', 0.5)
                actual_correct = self._agent_in_consensus(agent.name, ctx.result)

                self.deps.calibration_tracker.record_prediction(
                    agent_name=agent.name,
                    predicted_confidence=predicted_confidence,
                    actual_correct=actual_correct,
                    domain=ctx.domain,
                )
            self._log(f"  [calibration] Recorded predictions for {len(ctx.agents)} agents")
        except Exception as e:
            self._log(f"  [calibration] Recording failed: {e}")

    def _record_suggestion_feedback(self, ctx: ProcessingContext) -> None:
        """Record suggestion feedback for audience learning (P10: SuggestionFeedback)."""
        if not self.deps.suggestion_feedback:
            return

        try:
            # Check for audience suggestions that influenced the debate
            if hasattr(ctx.result, 'audience_contributions') and ctx.result.audience_contributions:
                for contrib in ctx.result.audience_contributions:
                    was_incorporated = contrib.get('incorporated', False)
                    self.deps.suggestion_feedback.record_feedback(
                        debate_id=ctx.debate_id,
                        suggestion_id=contrib.get('id', ''),
                        incorporated=was_incorporated,
                    )
                self._log(f"  [feedback] Recorded {len(ctx.result.audience_contributions)} suggestions")
        except Exception as e:
            self._log(f"  [feedback] Recording failed: {e}")

    async def _extract_insights(self, ctx: ProcessingContext) -> None:
        """Extract and store insights for pattern learning (P2: InsightExtractor)."""
        if not self.deps.insight_store or not self.deps.insight_available:
            return

        try:
            if ctx.result.consensus_reached and ctx.result.final_answer:
                await self.deps.insight_store.extract_and_store(
                    debate_result=ctx.result,
                    topic=ctx.topic,
                    agents=[a.name for a in ctx.agents],
                )
                self._log(f"  [insights] Extracted insights from debate")
        except Exception as e:
            self._log(f"  [insights] Extraction failed: {e}")

    async def _record_agent_memories(self, ctx: ProcessingContext) -> None:
        """Record agent memories for cumulative learning (P3: MemoryStream)."""
        if not self.deps.memory_stream:
            return

        try:
            for agent in ctx.agents:
                agent_msgs = [
                    m for m in getattr(ctx.result, 'messages', [])
                    if getattr(m, 'agent', '') == agent.name
                ]
                if agent_msgs:
                    await self.deps.memory_stream.record(
                        agent_name=agent.name,
                        debate_id=ctx.debate_id,
                        messages=agent_msgs,
                        outcome='consensus' if ctx.result.consensus_reached else 'no_consensus',
                    )
            self._log(f"  [memory] Recorded memories for {len(ctx.agents)} agents")
        except Exception as e:
            self._log(f"  [memory] Recording failed: {e}")

    def _record_persona_performance(self, ctx: ProcessingContext) -> None:
        """Record persona performance metrics (P8: PersonaManager)."""
        if not self.deps.persona_manager:
            return

        try:
            for agent in ctx.agents:
                in_consensus = self._agent_in_consensus(agent.name, ctx.result)
                self.deps.persona_manager.record_performance(
                    agent_name=agent.name,
                    topic=ctx.topic,
                    contributed_to_consensus=in_consensus,
                    domain=ctx.domain,
                )
            self._log(f"  [persona] Recorded performance for {len(ctx.agents)} agents")
        except Exception as e:
            self._log(f"  [persona] Recording failed: {e}")

    async def _extract_patterns(self, ctx: ProcessingContext) -> None:
        """Extract and store winning argument patterns (P9: PromptEvolver)."""
        if not self.deps.pattern_store:
            return

        try:
            if ctx.result.consensus_reached:
                patterns = self._identify_winning_patterns(ctx.result)
                for pattern in patterns:
                    await self.deps.pattern_store.store(pattern)
                if patterns:
                    self._log(f"  [patterns] Extracted {len(patterns)} winning patterns")
        except Exception as e:
            self._log(f"  [patterns] Extraction failed: {e}")

    def _identify_winning_patterns(self, result: Any) -> List[dict]:
        """Identify argument patterns that led to consensus."""
        patterns = []
        if not hasattr(result, 'messages'):
            return patterns

        # Look for messages that received positive critiques or votes
        for msg in result.messages:
            if hasattr(msg, 'critique_score') and msg.critique_score and msg.critique_score > 0.7:
                patterns.append({
                    'type': 'high_critique_score',
                    'content_snippet': str(msg.content)[:200] if msg.content else '',
                    'agent': getattr(msg, 'agent', 'unknown'),
                    'score': msg.critique_score,
                })

        return patterns

    def _analyze_meta_critique(self, ctx: ProcessingContext) -> None:
        """Analyze debate process and store recommendations (P12: MetaCritiqueAnalyzer)."""
        # Meta-critique analysis is typically done inline
        # This hook allows for future extraction
        pass

    def _record_elo_match(self, ctx: ProcessingContext) -> None:
        """Update ELO ratings based on debate outcome (P13: EloSystem)."""
        if not self.deps.elo_system or not self.deps.elo_available:
            return

        try:
            if not ctx.result.consensus_reached:
                self._log(f"  [elo] Skipped - no consensus reached")
                return

            # Determine winner based on final answer attribution
            winner = None
            for agent in ctx.agents:
                if self._agent_in_consensus(agent.name, ctx.result):
                    winner = agent.name
                    break

            if winner:
                participants = [a.name for a in ctx.agents]
                self.deps.elo_system.record_match(
                    winner=winner,
                    participants=participants,
                    domain=ctx.domain,
                )
                self._log(f"  [elo] Recorded match, winner: {winner}")
        except Exception as e:
            self._log(f"  [elo] Recording failed: {e}")

    def _record_domain_calibration(self, ctx: ProcessingContext) -> None:
        """Record domain-specific calibration for grounded personas."""
        if not self.deps.elo_system or not self.deps.elo_available:
            return

        try:
            for agent in ctx.agents:
                agent_correct = self._agent_in_consensus(agent.name, ctx.result)
                confidence = getattr(ctx.result, 'confidence', 0.7)
                self.deps.elo_system.record_domain_prediction(
                    agent_name=agent.name,
                    domain=ctx.domain,
                    confidence=confidence,
                    correct=agent_correct,
                )
            self._log(f"  [calibration] Recorded domain predictions for {len(ctx.agents)} agents in '{ctx.domain}'")
        except Exception as e:
            self._log(f"  [calibration] Domain recording failed: {e}")

    def _record_positions(self, ctx: ProcessingContext) -> None:
        """Record positions to ledger for grounded personas."""
        if not self.deps.position_ledger or not self.deps.grounded_available:
            return

        try:
            messages = getattr(ctx.result, 'messages', [])
            recorded = 0
            for msg in messages:
                if hasattr(msg, 'agent') and hasattr(msg, 'content') and msg.content:
                    self.deps.position_ledger.record_position(
                        agent_name=msg.agent,
                        claim=str(msg.content)[:1000],
                        confidence=0.7,
                        debate_id=ctx.debate_id,
                        round_num=getattr(msg, 'round', 0),
                    )
                    recorded += 1

            # Resolve positions based on outcome
            outcome = "correct" if ctx.result.consensus_reached else "unresolved"
            for agent in ctx.agents:
                positions = self.deps.position_ledger.get_agent_positions(
                    agent_name=agent.name,
                    limit=10,
                )
                for pos in positions:
                    agent_outcome = "correct" if self._agent_in_consensus(agent.name, ctx.result) else "incorrect"
                    self.deps.position_ledger.resolve_position(pos.id, agent_outcome)

            self._log(f"  [grounded] Recorded {recorded} positions, resolved for {len(ctx.agents)} agents")
        except Exception as e:
            self._log(f"  [grounded] Position recording failed: {e}")

    def _update_relationships(self, ctx: ProcessingContext) -> None:
        """Update agent relationships based on debate dynamics."""
        if not self.deps.relationship_tracker or not self.deps.grounded_available:
            return

        try:
            participants = [a.name for a in ctx.agents]

            # Determine winner from votes
            winner = None
            votes = getattr(ctx.result, 'votes', [])
            if votes:
                vote_tally = {}
                for v in votes:
                    choice = getattr(v, 'choice', None)
                    if choice:
                        vote_tally[choice] = vote_tally.get(choice, 0) + 1
                if vote_tally:
                    winner = max(vote_tally.items(), key=lambda x: x[1])[0]

            # Extract critiques for relationship tracking
            critiques_data = []
            messages = getattr(ctx.result, 'messages', [])
            for msg in messages:
                critique = getattr(msg, 'critique', None)
                if critique:
                    critiques_data.append({
                        'critic': getattr(msg, 'agent', 'unknown'),
                        'target': getattr(critique, 'target', 'unknown'),
                        'accepted': getattr(critique, 'accepted', False),
                    })

            # Convert votes to dict format
            votes_dict = {}
            for v in votes:
                agent = getattr(v, 'agent', None)
                choice = getattr(v, 'choice', None)
                if agent and choice:
                    votes_dict[agent] = choice

            self.deps.relationship_tracker.update_from_debate(
                debate_id=ctx.debate_id,
                participants=participants,
                winner=winner,
                votes=votes_dict,
                critiques=critiques_data,
            )
            self._log(f"  [relationships] Updated for {len(participants)} participants")
        except Exception as e:
            self._log(f"  [relationships] Update failed: {e}")

    def _update_reputation(self, ctx: ProcessingContext) -> None:
        """Update agent reputation based on debate outcome."""
        if not self.deps.critique_store or not ctx.result.consensus_reached:
            return

        try:
            winning_proposal = ctx.result.final_answer or ""
            for agent in ctx.agents:
                proposal_accepted = agent.name.lower() in winning_proposal.lower()
                self.deps.critique_store.update_reputation(
                    agent.name,
                    proposal_made=True,
                    proposal_accepted=proposal_accepted,
                )
            self._log(f"  [reputation] Updated for {len(ctx.agents)} agents")
        except Exception as e:
            self._log(f"  [reputation] Update failed: {e}")

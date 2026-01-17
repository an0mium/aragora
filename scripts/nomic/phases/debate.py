"""
Debate phase for nomic loop.

Phase 1: Agents debate what to improve
- Multi-agent debate orchestration
- Learning context injection
- Post-debate analysis hooks
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import DebateResult


@dataclass
class DebateConfig:
    """Configuration for debate phase."""

    rounds: int = 2
    consensus_mode: str = "judge"
    judge_selection: str = "elo_ranked"
    proposer_count: int = 4
    role_rotation: bool = True
    asymmetric_stances: bool = False
    audience_injection: str = "summary"
    enable_research: bool = True


@dataclass
class LearningContext:
    """Aggregated learning context for debate."""

    failure_lessons: str = ""
    successful_patterns: str = ""
    failure_patterns: str = ""
    agent_reputations: str = ""
    continuum_patterns: str = ""
    stale_claims: str = ""
    introspection: str = ""
    consensus_history: str = ""
    evidence_context: str = ""
    insight_context: str = ""
    similar_debates: str = ""
    crux_context: str = ""
    meta_observations: str = ""
    relationship_context: str = ""
    calibration_context: str = ""
    pulse_context: str = ""
    # Audit integration
    audit_context: str = ""  # Findings from CodebaseAuditor
    audit_proposals: List[str] = field(default_factory=list)

    def to_string(self) -> str:
        """Combine all context into a single string."""
        parts = []
        for name, value in [
            ("failure_lessons", self.failure_lessons),
            ("successful_patterns", self.successful_patterns),
            ("failure_patterns", self.failure_patterns),
            ("stale_claims", self.stale_claims),
            ("agent_reputations", self.agent_reputations),
            ("continuum_patterns", self.continuum_patterns),
            ("introspection", self.introspection),
            ("consensus_history", self.consensus_history),
            ("evidence_context", self.evidence_context),
            ("insight_context", self.insight_context),
            ("similar_debates", self.similar_debates),
            ("crux_context", self.crux_context),
            ("meta_observations", self.meta_observations),
            ("relationship_context", self.relationship_context),
            ("calibration_context", self.calibration_context),
            ("pulse_context", self.pulse_context),
            ("audit_context", self.audit_context),
        ]:
            if value:
                parts.append(value)
        return "\n".join(parts)


@dataclass
class PostDebateHooks:
    """Hooks for post-debate processing."""

    on_consensus_stored: Optional[Callable] = None
    on_calibration_recorded: Optional[Callable] = None
    on_insights_extracted: Optional[Callable] = None
    on_memories_recorded: Optional[Callable] = None
    on_persona_recorded: Optional[Callable] = None
    on_patterns_extracted: Optional[Callable] = None
    on_meta_analyzed: Optional[Callable] = None
    on_elo_recorded: Optional[Callable] = None
    on_positions_recorded: Optional[Callable] = None
    on_relationships_updated: Optional[Callable] = None
    on_risks_tracked: Optional[Callable] = None
    on_claims_extracted: Optional[Callable] = None
    on_belief_network_built: Optional[Callable] = None


SAFETY_PREAMBLE = """SAFETY RULES:
1. Propose ADDITIONS, not removals
2. Build new capabilities, don't simplify existing ones
3. Check codebase analysis - if a feature exists, don't propose it
4. Learn from previous failures shown below"""


class DebatePhase:
    """
    Handles the improvement debate phase.

    Orchestrates multi-agent debate to determine what improvement
    to implement in the current cycle.
    """

    def __init__(
        self,
        aragora_path: Path,
        agents: List[Any],
        arena_factory: Callable[..., Any],
        environment_factory: Callable[..., Any],
        protocol_factory: Callable[..., Any],
        config: Optional[DebateConfig] = None,
        nomic_integration: Optional[Any] = None,
        cycle_count: int = 0,
        initial_proposal: Optional[str] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit_fn: Optional[Callable[..., None]] = None,
        record_replay_fn: Optional[Callable[..., None]] = None,
    ):
        """
        Initialize the debate phase.

        Args:
            aragora_path: Path to the aragora project root
            agents: List of agent instances for debate
            arena_factory: Factory to create Arena instances
            environment_factory: Factory to create Environment instances
            protocol_factory: Factory to create DebateProtocol instances
            config: Debate configuration
            nomic_integration: Optional NomicIntegration for probing/checkpointing
            cycle_count: Current cycle number
            initial_proposal: Optional human-submitted proposal
            log_fn: Function to log messages
            stream_emit_fn: Function to emit streaming events
            record_replay_fn: Function to record replay events
        """
        self.aragora_path = aragora_path
        self.agents = agents
        self._arena_factory = arena_factory
        self._environment_factory = environment_factory
        self._protocol_factory = protocol_factory
        self.config = config or DebateConfig()
        self.nomic_integration = nomic_integration
        self.cycle_count = cycle_count
        self.initial_proposal = initial_proposal
        self._log = log_fn or print
        self._stream_emit = stream_emit_fn or (lambda *args: None)
        self._record_replay = record_replay_fn or (lambda *args: None)

    async def execute(
        self,
        codebase_context: str = "",
        recent_changes: str = "",
        learning_context: Optional[LearningContext] = None,
        hooks: Optional[PostDebateHooks] = None,
        arena_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DebateResult:
        """
        Execute the debate phase.

        Args:
            codebase_context: Context about current codebase features
            recent_changes: Recent changes to the codebase
            learning_context: Aggregated learning context
            hooks: Post-debate processing hooks
            arena_kwargs: Additional kwargs for Arena creation

        Returns:
            DebateResult with debate outcome
        """
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 1: IMPROVEMENT DEBATE")
        self._log("=" * 70)
        self._stream_emit(
            "on_phase_start", "debate", self.cycle_count, {"agents": len(self.agents)}
        )
        self._record_replay("phase", "system", "debate")

        # Load belief network from previous cycle
        await self._load_previous_belief_network()

        # Build task prompt
        task = self._build_task_prompt(
            codebase_context,
            recent_changes,
            learning_context or LearningContext(),
        )

        # Create environment and protocol
        context_section = self._build_context_section(codebase_context)
        env = self._environment_factory(task=task, context=context_section)

        # Enable asymmetric stances periodically
        asymmetric = self.cycle_count % 15 == 0
        if asymmetric:
            self._log("  [stances] Devil's advocate mode enabled")

        protocol = self._protocol_factory(
            rounds=self.config.rounds,
            consensus=self.config.consensus_mode,
            judge_selection=self.config.judge_selection,
            proposer_count=self.config.proposer_count,
            role_rotation=self.config.role_rotation,
            asymmetric_stances=asymmetric,
            rotate_stances=asymmetric,
            audience_injection=self.config.audience_injection,
            enable_research=self.config.enable_research,
        )

        # Probe agents for reliability
        agent_weights = await self._probe_agents()

        # Create arena
        arena = self._arena_factory(
            env,
            self.agents,
            protocol,
            agent_weights=agent_weights,
            **(arena_kwargs or {}),
        )

        # Run debate
        result = await self._run_debate(arena)

        # Process post-debate hooks
        if hooks:
            await self._run_post_debate_hooks(result, hooks)

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end",
            "debate",
            self.cycle_count,
            result.consensus_reached,
            phase_duration,
            {"confidence": result.confidence},
        )

        return DebateResult(
            success=result.consensus_reached,
            data={
                "final_answer": result.final_answer,
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
            },
            duration_seconds=phase_duration,
            improvement=result.final_answer or "",
            consensus_reached=result.consensus_reached,
            confidence=result.confidence,
            votes=[(v.agent, v.choice) for v in (result.votes or [])],
        )

    async def _load_previous_belief_network(self) -> None:
        """Load belief network from previous cycle if available."""
        if not self.nomic_integration or self.cycle_count <= 1:
            return

        try:
            prev_cycle = self.cycle_count - 1
            checkpoints = await self.nomic_integration.list_checkpoints()
            for cp in checkpoints:
                if f"cycle_{prev_cycle}_debate" in str(cp):
                    self._log(f"  [belief] Loading belief network from cycle {prev_cycle}")
                    await self.nomic_integration.resume_from_checkpoint(cp.get("checkpoint_id", ""))
                    break
        except Exception as e:
            self._log(f"  [belief] Failed to load previous belief network: {e}")

    def _build_task_prompt(
        self,
        codebase_context: str,
        recent_changes: str,
        learning: LearningContext,
    ) -> str:
        """Build the debate task prompt."""
        initial_section = ""
        if self.initial_proposal:
            initial_section = f"""

===== HUMAN-SUBMITTED PROPOSAL =====
A human has submitted the following proposal for your consideration.
You may adopt it, critique it, improve upon it, or propose something entirely different.

{self.initial_proposal}
====================================
"""
            self._log(f"  Including human proposal: {self.initial_proposal[:100]}...")

        # Build audit proposals section (high-priority)
        audit_section = ""
        if learning.audit_proposals:
            proposals_text = "\n".join(f"  - {p}" for p in learning.audit_proposals[:5])
            audit_section = f"""

===== AUDIT-IDENTIFIED ISSUES (High Priority) =====
Automated codebase analysis has identified the following issues.
These are VERIFIED problems that should be considered as high-priority candidates:

{proposals_text}

Consider addressing these issues OR propose something more impactful.
===================================================
"""
            self._log(f"  Including {len(learning.audit_proposals)} audit proposals")

        learning_text = learning.to_string()

        return f"""{SAFETY_PREAMBLE}

What single improvement would most benefit aragora RIGHT NOW?

CRITICAL: Read the codebase analysis below carefully. DO NOT propose features that already exist.
{learning_text}
{audit_section}
Consider what would make aragora:
- More INTERESTING (novel, creative, intellectually stimulating)
- More POWERFUL (capable, versatile, effective)
- More VIRAL (shareable, demonstrable, meme-worthy)
- More USEFUL (practical, solves real problems)
{initial_section}
Each agent should propose ONE specific, implementable feature.
Be concrete: describe what it does, how it works, and why it matters.
After debate, reach consensus on THE SINGLE BEST improvement to implement this cycle.

Recent changes:
{recent_changes}"""

    def _build_context_section(self, codebase_context: str) -> str:
        """Build the context section for the environment."""
        if codebase_context and len(codebase_context) > 500:
            return f"""
===== CODEBASE ANALYSIS (from Claude + Codex who explored the code) =====
The following is a comprehensive analysis of aragora's EXISTING features.
DO NOT propose features that already exist below.

{codebase_context}
========================================================================"""
        return f"Current aragora features:\n{codebase_context}"

    async def _probe_agents(self) -> Dict[str, float]:
        """Probe agents for reliability weights."""
        if not self.nomic_integration:
            return {}

        try:
            self._log("  [integration] Probing debate agents for reliability...")
            weights = await self.nomic_integration.probe_agents(
                self.agents,
                probe_count=2,
                min_weight=0.5,
            )
            reliable = sum(1 for w in weights.values() if w >= 0.7)
            self._log(f"  [integration] Agent weights: {reliable}/{len(self.agents)} reliable")
            return weights
        except Exception as e:
            self._log(f"  [integration] Probing failed: {e}")
            return {}

    async def _run_debate(self, arena: Any) -> Any:
        """Run the debate and return result."""
        try:
            result = await arena.run()
            return result
        except Exception as e:
            self._log(f"  Debate error: {e}")
            raise

    async def _run_post_debate_hooks(
        self,
        result: Any,
        hooks: PostDebateHooks,
    ) -> None:
        """Run post-debate processing hooks."""
        topic = self.initial_proposal[:200] if self.initial_proposal else "improvement"

        try:
            if hooks.on_consensus_stored and result.consensus_reached:
                await self._safe_call(hooks.on_consensus_stored, result, topic)

            if hooks.on_calibration_recorded:
                await self._safe_call(hooks.on_calibration_recorded, result, self.agents)

            if hooks.on_insights_extracted:
                await self._safe_call(hooks.on_insights_extracted, result)

            if hooks.on_memories_recorded:
                await self._safe_call(hooks.on_memories_recorded, result, topic)

            if hooks.on_persona_recorded:
                await self._safe_call(hooks.on_persona_recorded, result, topic)

            if hooks.on_patterns_extracted:
                await self._safe_call(hooks.on_patterns_extracted, result)

            if hooks.on_meta_analyzed:
                await self._safe_call(hooks.on_meta_analyzed, result)

            if hooks.on_elo_recorded:
                await self._safe_call(hooks.on_elo_recorded, result, topic)

            if hooks.on_positions_recorded:
                await self._safe_call(hooks.on_positions_recorded, result)

            if hooks.on_relationships_updated:
                await self._safe_call(hooks.on_relationships_updated, result, self.agents)

            if hooks.on_risks_tracked:
                await self._safe_call(hooks.on_risks_tracked, result, topic)

            if hooks.on_claims_extracted:
                await self._safe_call(hooks.on_claims_extracted, result)

            if hooks.on_belief_network_built:
                await self._safe_call(hooks.on_belief_network_built, result)

        except Exception as e:
            self._log(f"  Post-debate hooks error: {e}")

    async def _safe_call(self, fn: Callable, *args) -> None:
        """Safely call a function, handling both sync and async."""
        try:
            result = fn(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            self._log(f"  Hook error: {e}")


__all__ = ["DebatePhase", "DebateConfig", "LearningContext", "PostDebateHooks"]

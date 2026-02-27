"""Unified Pipeline Orchestrator.

Composes all Wave 1-4 modules into a single execution path:

    prompt → extend → research → debate (diverse team) → plan → execute → feedback → ELO

This is the "one function" entry point that wires together:
- InputExtensionEngine (enriches user prompts)
- UnifiedResearcher (gathers context from KM/Obsidian/web)
- ProviderDiversityFilter (ensures multi-provider debate teams)
- Arena (debate orchestration)
- DecisionPlanFactory (creates execution plans from debate)
- AutonomyGate (human-in-the-loop controls)
- PlanExecutor (executes approved plans)
- OutcomeFeedbackRecorder (records outcomes to KM/ELO/calibrator)
- PhaseELOTracker (phase-tagged agent ratings)
- MetaLoopTrigger (self-improvement detection)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the unified pipeline orchestrator."""

    # User preset (founder, cto, team, non_technical)
    preset_name: str = "cto"

    # Domain context for input extension
    domain: str = ""

    # Debate settings (overridable, defaults come from preset)
    debate_rounds: int | None = None
    agent_count: int | None = None
    consensus_threshold: float | None = None

    # Autonomy
    autonomy_level: str = "propose_and_approve"

    # Provider diversity
    min_providers: int = 2

    # Self-improvement
    enable_meta_loop: bool = False

    # Execution
    execution_mode: str = "workflow"
    skip_execution: bool = False


@dataclass
class OrchestratorResult:
    """Result from a unified pipeline run."""

    run_id: str
    prompt: str

    # Stage outputs
    extended_input: Any | None = None
    research_context: Any | None = None
    diversity_report: Any | None = None
    debate_result: Any | None = None
    decision_plan: Any | None = None
    plan_outcome: Any | None = None
    pipeline_outcome: Any | None = None
    meta_loop_result: Any | None = None

    # Tracking
    stages_completed: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    approvals_needed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        return len(self.errors) == 0 and "debate" in self.stages_completed

    @property
    def quality_score(self) -> float:
        if self.pipeline_outcome is not None and hasattr(
            self.pipeline_outcome, "overall_quality_score"
        ):
            return self.pipeline_outcome.overall_quality_score
        return 0.0


class UnifiedOrchestrator:
    """Composes Wave 1-4 modules into a single execution path.

    All dependencies are optional — the orchestrator gracefully degrades
    when components are unavailable. This allows incremental adoption:
    pass only what you have, and the orchestrator uses what it gets.
    """

    def __init__(
        self,
        # Wave 1: Truth-seeking
        input_extension: Any | None = None,
        researcher: Any | None = None,
        diversity_filter: Any | None = None,
        # Wave 2: Learning
        elo_tracker: Any | None = None,
        calibrator: Any | None = None,
        feedback_recorder: Any | None = None,
        # Wave 3: Self-improvement
        meta_loop: Any | None = None,
        # Existing infrastructure
        arena_factory: Any | None = None,
        plan_factory: Any | None = None,
        plan_executor: Any | None = None,
        knowledge_mound: Any | None = None,
    ) -> None:
        self._input_extension = input_extension
        self._researcher = researcher
        self._diversity_filter = diversity_filter
        self._elo_tracker = elo_tracker
        self._calibrator = calibrator
        self._feedback_recorder = feedback_recorder
        self._meta_loop = meta_loop
        self._arena_factory = arena_factory
        self._plan_factory = plan_factory
        self._plan_executor = plan_executor
        self._km = knowledge_mound

    async def run(
        self,
        prompt: str,
        config: OrchestratorConfig | None = None,
        agents: list[Any] | None = None,
        approval_callback: Any | None = None,
    ) -> OrchestratorResult:
        """Execute the full pipeline from prompt to outcome.

        Args:
            prompt: User's natural language input.
            config: Pipeline configuration (defaults to CTO preset).
            agents: Pre-selected agents for debate (optional).
            approval_callback: Called when human approval is needed.
                Signature: async (stage: str, artifact: Any) -> bool

        Returns:
            OrchestratorResult with outputs from each stage.
        """
        cfg = config or OrchestratorConfig()
        result = OrchestratorResult(run_id=str(uuid.uuid4()), prompt=prompt)
        start = time.monotonic()

        # Load preset defaults
        preset_config = self._load_preset(cfg.preset_name)

        # Merge preset with explicit overrides
        debate_rounds = cfg.debate_rounds or preset_config.get("debate", {}).get("rounds", 3)
        agent_count = cfg.agent_count or preset_config.get("debate", {}).get("agent_count", 5)
        consensus_threshold = cfg.consensus_threshold or preset_config.get("debate", {}).get(
            "consensus_threshold", 0.6
        )

        # Build autonomy gates
        gates = self._build_gates(cfg.autonomy_level)

        # --- Stage 1: Research ---
        try:
            result.research_context = await self._do_research(prompt)
            result.stages_completed.append("research")
        except Exception:
            logger.warning("Research stage failed, continuing without context")
            result.stages_skipped.append("research")

        # --- Stage 2: Extend Input ---
        try:
            result.extended_input = await self._do_extend(
                prompt, cfg.domain, result.research_context
            )
            result.stages_completed.append("extend")
        except Exception:
            logger.warning("Input extension failed, using raw prompt")
            result.stages_skipped.append("extend")

        # --- Stage 3: Debate ---
        try:
            debate_prompt = prompt
            if result.extended_input is not None and hasattr(
                result.extended_input, "to_context_block"
            ):
                context_block = result.extended_input.to_context_block()
                if context_block:
                    debate_prompt = f"{prompt}\n\n{context_block}"

            # Apply diversity filter to agents
            debate_agents = agents
            if debate_agents and self._diversity_filter is not None:
                debate_agents, report = self._diversity_filter.enforce(debate_agents)
                result.diversity_report = report

            result.debate_result = await self._do_debate(
                debate_prompt,
                debate_agents,
                rounds=debate_rounds,
                agent_count=agent_count,
                consensus_threshold=consensus_threshold,
            )
            result.stages_completed.append("debate")

            # Update phase ELO from debate
            if self._elo_tracker is not None and result.debate_result is not None:
                self._update_phase_elo(result.debate_result, cfg.domain)

        except Exception as exc:
            logger.error("Debate stage failed: %s", exc)
            result.errors.append(f"Debate failed: {exc}")
            result.duration_s = time.monotonic() - start
            return result

        # --- Stage 4: Create Decision Plan ---
        if result.debate_result is not None and self._plan_factory is not None:
            try:
                result.decision_plan = self._plan_factory.from_debate_result(result.debate_result)
                result.stages_completed.append("plan")
            except Exception:
                logger.warning("Plan creation failed")
                result.stages_skipped.append("plan")

        # --- Stage 5: Approval Gate ---
        if result.decision_plan is not None and gates.get("spec") is not None:
            gate = gates["spec"]
            if gate.needs_approval():
                if approval_callback is not None:
                    approved = await approval_callback("spec", result.decision_plan)
                    if not approved:
                        result.approvals_needed.append("spec")
                        result.duration_s = time.monotonic() - start
                        return result
                else:
                    result.approvals_needed.append("spec")
                    result.duration_s = time.monotonic() - start
                    return result

        # --- Stage 6: Execute Plan ---
        if (
            result.decision_plan is not None
            and self._plan_executor is not None
            and not cfg.skip_execution
        ):
            try:
                result.plan_outcome = await self._plan_executor.execute(
                    result.decision_plan,
                    execution_mode=cfg.execution_mode,
                )
                result.stages_completed.append("execute")
            except Exception:
                logger.warning("Execution failed")
                result.stages_skipped.append("execute")

        # --- Stage 7: Record Outcome ---
        if self._feedback_recorder is not None and result.debate_result is not None:
            try:
                outcome = self._build_outcome(result, cfg)
                self._feedback_recorder.record(outcome)
                result.pipeline_outcome = outcome
                result.stages_completed.append("feedback")
            except Exception:
                logger.warning("Feedback recording failed")
                result.stages_skipped.append("feedback")

        # --- Stage 8: Meta-Loop Check ---
        if cfg.enable_meta_loop and self._meta_loop is not None:
            try:
                self._meta_loop.increment_cycle()
                if self._meta_loop.should_trigger():
                    targets = self._meta_loop.identify_targets()
                    result.meta_loop_result = self._meta_loop.execute(targets)
                result.stages_completed.append("meta_loop")
            except Exception:
                logger.warning("Meta-loop check failed")
                result.stages_skipped.append("meta_loop")

        result.duration_s = time.monotonic() - start
        return result

    def _load_preset(self, preset_name: str) -> dict[str, Any]:
        """Load a user preset configuration."""
        try:
            from aragora.pipeline.user_presets import get_preset

            preset = get_preset(preset_name)
            return preset.to_pipeline_config()
        except (ImportError, ValueError):
            return {}

    def _build_gates(self, autonomy_level: str) -> dict[str, Any]:
        """Build autonomy gates for the pipeline."""
        try:
            from aragora.pipeline.autonomy import AutonomyLevel, create_gates

            level = AutonomyLevel.from_string(autonomy_level)
            return create_gates(level)
        except (ImportError, ValueError):
            return {}

    async def _do_research(self, prompt: str) -> Any:
        """Run research phase."""
        if self._researcher is None:
            return None
        return await self._researcher.research(prompt)

    async def _do_extend(self, prompt: str, domain: str, research_context: Any) -> Any:
        """Run input extension phase."""
        if self._input_extension is None:
            return None
        return await self._input_extension.extend(
            prompt, domain=domain, research_context=research_context
        )

    async def _do_debate(
        self,
        prompt: str,
        agents: list[Any] | None,
        rounds: int,
        agent_count: int,
        consensus_threshold: float,
    ) -> Any:
        """Run debate phase."""
        if self._arena_factory is not None:
            return await self._arena_factory(
                prompt,
                agents=agents,
                rounds=rounds,
                agent_count=agent_count,
                consensus_threshold=consensus_threshold,
            )
        return None

    def _update_phase_elo(self, debate_result: Any, domain: str) -> None:
        """Update phase ELO ratings from debate results."""
        if not hasattr(debate_result, "participants"):
            return
        domain_key = domain or "general"
        for participant in debate_result.participants:
            name = participant if isinstance(participant, str) else str(participant)
            won = hasattr(debate_result, "final_answer") and debate_result.final_answer
            self._elo_tracker.record_match(
                agent_name=name,
                domain=domain_key,
                phase="debate",
                won=won,
            )

    def _build_outcome(self, result: OrchestratorResult, cfg: OrchestratorConfig) -> Any:
        """Build a PipelineOutcome from the orchestrator result."""
        from aragora.pipeline.outcome_feedback import PipelineOutcome

        outcome = PipelineOutcome(
            pipeline_id=result.run_id,
            run_type="user_project",
            domain=cfg.domain or "general",
            execution_succeeded="execute" in result.stages_completed,
        )

        if result.plan_outcome is not None:
            if hasattr(result.plan_outcome, "tests_passed"):
                outcome.tests_passed = result.plan_outcome.tests_passed
            if hasattr(result.plan_outcome, "tests_failed"):
                outcome.tests_failed = result.plan_outcome.tests_failed
            if hasattr(result.plan_outcome, "files_changed"):
                outcome.files_changed = result.plan_outcome.files_changed

        outcome.total_duration_s = result.duration_s
        return outcome

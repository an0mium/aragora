"""Prompt Conductor - Orchestrates the full vague-prompt-to-specification pipeline.

Coordinates decomposition, interrogation, research, and specification building
into a single coherent flow with configurable autonomy levels.

Usage:
    conductor = PromptConductor()
    spec = await conductor.run("I want to improve the onboarding flow")

    # With user interaction callback
    conductor = PromptConductor(
        on_questions=my_question_handler,
        autonomy=AutonomyLevel.PROPOSE_AND_APPROVE,
    )
    spec = await conductor.run("Make the app faster")
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aragora.prompt_engine.decomposer import PromptDecomposer
from aragora.prompt_engine.interrogator import PromptInterrogator
from aragora.prompt_engine.researcher import PromptResearcher
from aragora.prompt_engine.spec_builder import SpecBuilder
from aragora.prompt_engine.timing import (
    PROMPT_ENGINE_TARGET_DURATION_MS,
    PipelineTiming,
    elapsed_ms,
    format_timings,
    start_timer,
)
from aragora.prompt_engine.types import (
    PROFILE_DEFAULTS,
    AutonomyLevel,
    ClarifyingQuestion,
    InterrogationDepth,
    PromptIntent,
    ResearchReport,
    Specification,
    UserProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class ConductorConfig:
    """Configuration for the PromptConductor."""

    autonomy: AutonomyLevel = AutonomyLevel.PROPOSE_AND_APPROVE
    interrogation_depth: InterrogationDepth = InterrogationDepth.THOROUGH
    auto_execute_threshold: float = 0.9
    skip_research: bool = False
    skip_interrogation: bool = False
    latency_target_ms: float = PROMPT_ENGINE_TARGET_DURATION_MS

    @classmethod
    def from_profile(cls, profile: UserProfile | str) -> ConductorConfig:
        """Create config from a user profile."""
        if isinstance(profile, str):
            try:
                profile = UserProfile(profile)
            except ValueError:
                profile = UserProfile.FOUNDER

        defaults = PROFILE_DEFAULTS.get(profile.value, {})
        return cls(
            autonomy=defaults.get("autonomy_level", AutonomyLevel.PROPOSE_AND_APPROVE),
            interrogation_depth=defaults.get("interrogation_depth", InterrogationDepth.THOROUGH),
            auto_execute_threshold=defaults.get("auto_execute_threshold", 0.9),
        )


@dataclass
class ConductorResult:
    """Result of a full conductor run."""

    specification: Specification
    intent: PromptIntent
    questions: list[ClarifyingQuestion] = field(default_factory=list)
    research: ResearchReport | None = None
    auto_approved: bool = False
    stages_completed: list[str] = field(default_factory=list)
    timing: PipelineTiming = field(default_factory=PipelineTiming)


# Type alias for the question handler callback
QuestionHandler = Callable[[list[ClarifyingQuestion]], Awaitable[list[ClarifyingQuestion]]]


class PromptConductor:
    """Orchestrates the full prompt-to-specification pipeline.

    The conductor coordinates four stages:
    1. Decompose: Parse vague prompt into structured intent
    2. Interrogate: Generate and resolve clarifying questions
    3. Research: Investigate context and current state
    4. Specify: Build formal implementation specification
    """

    def __init__(
        self,
        config: ConductorConfig | None = None,
        on_questions: QuestionHandler | None = None,
        knowledge_mound: Any | None = None,
        agent: Any | None = None,
    ) -> None:
        self._config = config or ConductorConfig()
        self._on_questions = on_questions
        self._km = knowledge_mound

        self._decomposer = PromptDecomposer(agent=agent, knowledge_mound=knowledge_mound)
        self._interrogator = PromptInterrogator(agent=agent)
        self._researcher = PromptResearcher(agent=agent, knowledge_mound=knowledge_mound)
        self._spec_builder = SpecBuilder(agent=agent)

    async def run(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> ConductorResult:
        """Run the full prompt-to-specification pipeline.

        Args:
            prompt: Raw user input (any vagueness level)
            context: Optional additional context

        Returns:
            ConductorResult with specification and full provenance
        """
        stages: list[str] = []
        stage_durations_ms: dict[str, float] = {}
        operation_timings = []
        pipeline_start = start_timer()

        # Stage 1: Decompose
        logger.info("Conductor: decomposing prompt")
        stage_start = start_timer()
        intent = await self._decomposer.decompose(prompt, context)
        stage_durations_ms["decompose"] = elapsed_ms(stage_start)
        operation_timings.extend(self._decomposer.last_operation_timings)
        stages.append("decompose")

        # Stage 2: Interrogate (if not skipped)
        questions: list[ClarifyingQuestion] = []
        if not self._config.skip_interrogation and intent.needs_clarification:
            logger.info("Conductor: generating clarifying questions")
            stage_start = start_timer()
            questions = await self._interrogator.interrogate(
                intent, depth=self._config.interrogation_depth
            )
            stage_durations_ms["interrogate"] = elapsed_ms(stage_start)
            operation_timings.extend(self._interrogator.last_operation_timings)
            stages.append("interrogate")

            # If we have a question handler, let the user answer
            if self._on_questions and questions:
                questions = await self._on_questions(questions)

            # In full auto mode, use defaults for unanswered questions
            if self._config.autonomy == AutonomyLevel.FULL_AUTO:
                for q in questions:
                    if not q.is_answered and q.default:
                        q.answer = q.default

        # Stage 3: Research (if not skipped)
        research: ResearchReport | None = None
        if not self._config.skip_research:
            logger.info("Conductor: researching context")
            stage_start = start_timer()
            research = await self._researcher.research(
                intent,
                answered_questions=questions,
                context=context,
            )
            stage_durations_ms["research"] = elapsed_ms(stage_start)
            operation_timings.extend(self._researcher.last_operation_timings)
            stages.append("research")

        # Stage 4: Build specification
        logger.info("Conductor: building specification")
        stage_start = start_timer()
        spec = await self._spec_builder.build(
            intent,
            answered_questions=questions,
            research=research,
            context=context,
        )
        stage_durations_ms["specify"] = elapsed_ms(stage_start)
        operation_timings.extend(self._spec_builder.last_operation_timings)
        stages.append("specify")

        # Determine if auto-approved
        auto_approved = (
            self._config.autonomy == AutonomyLevel.FULL_AUTO
            and spec.confidence >= self._config.auto_execute_threshold
        )

        timing = PipelineTiming(
            total_duration_ms=elapsed_ms(pipeline_start),
            stage_durations_ms=stage_durations_ms,
            operation_timings=operation_timings,
            target_duration_ms=self._config.latency_target_ms,
        )

        logger.info(
            "Conductor latency total=%.1fms target=%.1fms stages=%s top_ops=%s",
            timing.total_duration_ms,
            timing.target_duration_ms,
            ", ".join(f"{name}={duration:.1f}ms" for name, duration in stage_durations_ms.items()),
            format_timings(timing.top_operations(limit=4)),
        )
        if not timing.is_within_target:
            logger.warning(
                "Conductor exceeded latency target by %.1fms bottlenecks=%s",
                timing.overrun_ms,
                format_timings(timing.bottlenecks()),
            )

        return ConductorResult(
            specification=spec,
            intent=intent,
            questions=questions,
            research=research,
            auto_approved=auto_approved,
            stages_completed=stages,
            timing=timing,
        )

    async def decompose_only(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> PromptIntent:
        """Run only the decomposition stage."""
        return await self._decomposer.decompose(prompt, context)

    async def interrogate_only(
        self,
        intent: PromptIntent,
        depth: InterrogationDepth | None = None,
    ) -> list[ClarifyingQuestion]:
        """Run only the interrogation stage."""
        return await self._interrogator.interrogate(
            intent,
            depth=depth or self._config.interrogation_depth,
        )

    async def research_only(
        self,
        intent: PromptIntent,
        questions: list[ClarifyingQuestion] | None = None,
    ) -> ResearchReport:
        """Run only the research stage."""
        return await self._researcher.research(intent, questions)

    async def specify_only(
        self,
        intent: PromptIntent,
        questions: list[ClarifyingQuestion] | None = None,
        research: ResearchReport | None = None,
    ) -> Specification:
        """Run only the specification stage."""
        return await self._spec_builder.build(intent, questions, research)

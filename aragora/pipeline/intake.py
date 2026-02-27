"""Pipeline intake — accept free-text prompts and feed into the pipeline.

This is the primary entry point for the "accept a vague prompt → structured
execution" flow.  It bridges ``BrainDumpParser`` (text normalisation),
``PipelineInterrogator`` (clarifying questions), and
``IdeaToExecutionPipeline`` (stage-by-stage execution).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Autonomy levels
# ---------------------------------------------------------------------------


class AutonomyLevel(IntEnum):
    """How much human oversight the pipeline requires.

    Higher values = more autonomous.
    """

    PROPOSE_AND_EXPLAIN = 1  # Show every decision, require approval per stage
    PROPOSE_AND_APPROVE = 2  # Show summary, require approval at stage boundaries
    EXECUTE_AND_REPORT = 3  # Run autonomously, report results with rollback option
    FULLY_AUTONOMOUS = 4  # Run end-to-end, notify on completion
    CONTINUOUS = 5  # Monitor for new inputs, execute on triggers


# ---------------------------------------------------------------------------
# Intake request / result
# ---------------------------------------------------------------------------


@dataclass
class IntakeRequest:
    """Incoming prompt from a user."""

    prompt: str
    autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_AND_APPROVE
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skip_interrogation: bool = False
    max_interrogation_turns: int = 5
    user_id: str | None = None
    workspace_id: str = "default"
    event_callback: Any | None = None


@dataclass
class IntakeResult:
    """Result of the intake process (before full pipeline execution)."""

    pipeline_id: str
    ideas: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    urgency_signals: list[str] = field(default_factory=list)
    interrogation_summary: str = ""
    refined_goal: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    ready_for_pipeline: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "ideas": self.ideas,
            "themes": self.themes,
            "urgency_signals": self.urgency_signals,
            "interrogation_summary": self.interrogation_summary,
            "refined_goal": self.refined_goal,
            "acceptance_criteria": self.acceptance_criteria,
            "ready_for_pipeline": self.ready_for_pipeline,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Intake engine
# ---------------------------------------------------------------------------


class PipelineIntake:
    """Accept a vague prompt and prepare it for pipeline execution.

    Usage::

        intake = PipelineIntake()
        result = await intake.process(IntakeRequest(prompt="improve error handling"))
        if result.ready_for_pipeline:
            pipeline_result = await intake.execute(result)
    """

    def __init__(self) -> None:
        self._parser: Any | None = None
        self._pipeline: Any | None = None

    # -- lazy imports to avoid heavy module loading at import time ----------

    def _get_parser(self) -> Any:
        if self._parser is None:
            from aragora.pipeline.brain_dump_parser import BrainDumpParser

            self._parser = BrainDumpParser()
        return self._parser

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

            self._pipeline = IdeaToExecutionPipeline()
        return self._pipeline

    # -- public API ---------------------------------------------------------

    async def process(
        self,
        request: IntakeRequest,
        *,
        input_fn: Any | None = None,
        print_fn: Any | None = None,
    ) -> IntakeResult:
        """Parse the prompt and optionally interrogate the user.

        Parameters
        ----------
        request:
            The intake request with prompt and configuration.
        input_fn:
            Callable for getting user input during interrogation.
            If ``None``, uses stdin (CLI) or must be provided for web.
        print_fn:
            Callable for displaying output during interrogation.
        """
        result = IntakeResult(pipeline_id=request.pipeline_id)

        # Step 1: Parse the raw prompt
        try:
            parser = self._get_parser()
            enriched = parser.parse_enriched(request.prompt)
            result.ideas = enriched.ideas
            result.themes = enriched.detected_themes
            result.urgency_signals = enriched.urgency_signals
            logger.info(
                "Parsed prompt into %d ideas with themes=%s",
                len(result.ideas),
                result.themes,
            )
        except Exception:
            logger.exception("Failed to parse prompt")
            # Fallback: treat entire prompt as single idea
            result.ideas = [request.prompt]

        # Step 2: Interrogation (unless skipped or fully autonomous)
        if (
            not request.skip_interrogation
            and request.autonomy_level <= AutonomyLevel.EXECUTE_AND_REPORT
        ):
            try:
                from aragora.pipeline.interrogator import PipelineInterrogator

                interrogator = PipelineInterrogator(
                    max_turns=request.max_interrogation_turns,
                )
                spec = await interrogator.interrogate(
                    initial_goal=request.prompt,
                    ideas=result.ideas,
                    input_fn=input_fn,
                    print_fn=print_fn,
                )
                result.refined_goal = spec.refined_goal or request.prompt
                result.acceptance_criteria = spec.acceptance_criteria
                result.interrogation_summary = spec.summary()
                logger.info(
                    "Interrogation complete: refined_goal=%r, criteria=%d",
                    result.refined_goal[:80],
                    len(result.acceptance_criteria),
                )
            except Exception:
                logger.exception("Interrogation failed, proceeding with raw ideas")
                result.refined_goal = request.prompt
        else:
            result.refined_goal = request.prompt

        result.ready_for_pipeline = len(result.ideas) > 0
        return result

    async def execute(
        self,
        intake_result: IntakeResult,
        request: IntakeRequest | None = None,
    ) -> Any:
        """Execute the full pipeline from intake results.

        Returns a ``PipelineResult`` from ``IdeaToExecutionPipeline``.
        """
        from aragora.pipeline.idea_to_execution import PipelineConfig

        autonomy = request.autonomy_level if request else AutonomyLevel.PROPOSE_AND_APPROVE

        config = PipelineConfig(
            stages_to_run=["ideation", "goals", "workflow", "orchestration"],
            human_approval_required=autonomy <= AutonomyLevel.PROPOSE_AND_APPROVE,
            enable_receipts=True,
            enable_km_persistence=True,
            event_callback=request.event_callback if request else None,
        )

        pipeline = self._get_pipeline()
        pipeline_result = pipeline.from_ideas(
            ideas=intake_result.ideas,
            auto_advance=autonomy >= AutonomyLevel.EXECUTE_AND_REPORT,
            pipeline_id=intake_result.pipeline_id,
            event_callback=config.event_callback,
        )

        logger.info(
            "Pipeline execution complete: pipeline_id=%s, stages=%s",
            intake_result.pipeline_id,
            list(pipeline_result.stage_status.keys())
            if hasattr(pipeline_result, "stage_status")
            else "unknown",
        )
        return pipeline_result

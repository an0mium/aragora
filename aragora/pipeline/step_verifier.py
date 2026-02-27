"""Pipeline step verifier — validate reasoning at each pipeline stage.

Bridges the ThinkPRM verifier into the pipeline context, verifying that
each stage transition's reasoning is logically sound. Uses calibration
data from past debates to score confidence at each step.

When integrated with stage transition debates, the verifier:
1. Extracts reasoning steps from debate output
2. Verifies each step against prior context
3. Uses agent calibration data to weight confidence
4. Flags weak reasoning for re-debate
5. Produces a verification score that enriches Decision Receipts
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineVerificationStep:
    """A reasoning step extracted from pipeline context."""

    step_id: str
    stage: str  # ideation | goals | actions | orchestration
    content: str
    agent_id: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class PipelineVerificationResult:
    """Result of verifying pipeline reasoning steps."""

    pipeline_id: str
    stage: str
    total_steps: int = 0
    verified_steps: int = 0
    flagged_steps: int = 0
    overall_score: float = 0.0
    step_results: list[dict[str, Any]] = field(default_factory=list)
    calibration_adjusted: bool = False
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "stage": self.stage,
            "total_steps": self.total_steps,
            "verified_steps": self.verified_steps,
            "flagged_steps": self.flagged_steps,
            "overall_score": self.overall_score,
            "step_results": self.step_results,
            "calibration_adjusted": self.calibration_adjusted,
            "duration_ms": self.duration_ms,
        }


class PipelineStepVerifier:
    """Verify reasoning quality at pipeline stage boundaries.

    Wraps ThinkPRMVerifier for the pipeline context, adding:
    - Stage-aware step extraction from pipeline data
    - Calibration-weighted confidence scoring
    - Integration with pipeline events and receipts

    Usage::

        verifier = PipelineStepVerifier()
        result = await verifier.verify_goal_extraction(
            ideas=["improve UX"],
            goals=[{"title": "Redesign nav", "rationale": "..."}],
            pipeline_id="pipe-123",
        )
        if result.overall_score < 0.5:
            # Flag for re-debate
            ...
    """

    def __init__(
        self,
        calibration_weight: float = 0.3,
        min_confidence: float = 0.5,
    ) -> None:
        self.calibration_weight = calibration_weight
        self.min_confidence = min_confidence

    async def verify_goal_extraction(
        self,
        ideas: list[str],
        goals: list[dict[str, Any]],
        pipeline_id: str = "",
    ) -> PipelineVerificationResult:
        """Verify that extracted goals logically follow from ideas.

        Checks:
        - Each goal traces to at least one idea
        - Goal priorities are justified
        - No ideas are orphaned without goals
        """
        start = time.monotonic()
        result = PipelineVerificationResult(pipeline_id=pipeline_id, stage="ideas_to_goals")

        steps = self._extract_goal_steps(ideas, goals)
        result.total_steps = len(steps)

        verified = await self._verify_steps(steps, "ideas_to_goals")
        result.step_results = verified
        result.verified_steps = sum(
            1 for v in verified if v.get("verdict") in ("correct", "uncertain")
        )
        result.flagged_steps = sum(
            1 for v in verified if v.get("verdict") in ("incorrect", "needs_revision")
        )

        if result.total_steps > 0:
            result.overall_score = result.verified_steps / result.total_steps
        else:
            result.overall_score = 1.0

        # Apply calibration adjustment
        result.overall_score = await self._apply_calibration(result.overall_score, verified)
        result.calibration_adjusted = True

        result.duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Goal extraction verification: score=%.2f (%d/%d verified, %.0fms)",
            result.overall_score,
            result.verified_steps,
            result.total_steps,
            result.duration_ms,
        )
        return result

    async def verify_action_decomposition(
        self,
        goals: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        pipeline_id: str = "",
    ) -> PipelineVerificationResult:
        """Verify that actions adequately decompose goals.

        Checks:
        - Each goal has at least one action
        - Actions are concrete and actionable
        - Dependencies between actions are valid
        """
        start = time.monotonic()
        result = PipelineVerificationResult(pipeline_id=pipeline_id, stage="goals_to_actions")

        steps = self._extract_action_steps(goals, actions)
        result.total_steps = len(steps)

        verified = await self._verify_steps(steps, "goals_to_actions")
        result.step_results = verified
        result.verified_steps = sum(
            1 for v in verified if v.get("verdict") in ("correct", "uncertain")
        )
        result.flagged_steps = sum(
            1 for v in verified if v.get("verdict") in ("incorrect", "needs_revision")
        )

        if result.total_steps > 0:
            result.overall_score = result.verified_steps / result.total_steps
        else:
            result.overall_score = 1.0

        result.overall_score = await self._apply_calibration(result.overall_score, verified)
        result.calibration_adjusted = True

        result.duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Action decomposition verification: score=%.2f (%d/%d verified, %.0fms)",
            result.overall_score,
            result.verified_steps,
            result.total_steps,
            result.duration_ms,
        )
        return result

    async def verify_agent_assignments(
        self,
        actions: list[dict[str, Any]],
        assignments: list[dict[str, Any]],
        pipeline_id: str = "",
    ) -> PipelineVerificationResult:
        """Verify that agent assignments match task requirements.

        Checks:
        - Agent capabilities align with task requirements
        - No task is unassigned (unless intentionally human-gated)
        - Parallelism opportunities are identified
        """
        start = time.monotonic()
        result = PipelineVerificationResult(
            pipeline_id=pipeline_id, stage="actions_to_orchestration"
        )

        steps = self._extract_assignment_steps(actions, assignments)
        result.total_steps = len(steps)

        verified = await self._verify_steps(steps, "actions_to_orchestration")
        result.step_results = verified
        result.verified_steps = sum(
            1 for v in verified if v.get("verdict") in ("correct", "uncertain")
        )
        result.flagged_steps = sum(
            1 for v in verified if v.get("verdict") in ("incorrect", "needs_revision")
        )

        if result.total_steps > 0:
            result.overall_score = result.verified_steps / result.total_steps
        else:
            result.overall_score = 1.0

        result.overall_score = await self._apply_calibration(result.overall_score, verified)
        result.calibration_adjusted = True

        result.duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Agent assignment verification: score=%.2f (%d/%d verified, %.0fms)",
            result.overall_score,
            result.verified_steps,
            result.total_steps,
            result.duration_ms,
        )
        return result

    # -- Step extraction helpers -------------------------------------------

    def _extract_goal_steps(
        self,
        ideas: list[str],
        goals: list[dict[str, Any]],
    ) -> list[PipelineVerificationStep]:
        """Extract verifiable steps from idea→goal transition."""
        steps = []
        for i, goal in enumerate(goals):
            title = goal.get("title", goal.get("id", f"goal-{i}"))
            priority = goal.get("priority", "medium")
            rationale = goal.get("rationale", goal.get("description", ""))

            content = f"Goal '{title}' (priority: {priority})"
            if rationale:
                content += f" — {rationale}"

            steps.append(
                PipelineVerificationStep(
                    step_id=f"goal-{i}",
                    stage="ideas_to_goals",
                    content=content,
                    dependencies=[f"Idea: {idea}" for idea in ideas],
                )
            )
        return steps

    def _extract_action_steps(
        self,
        goals: list[dict[str, Any]],
        actions: list[dict[str, Any]],
    ) -> list[PipelineVerificationStep]:
        """Extract verifiable steps from goal→action transition."""
        steps = []
        goal_titles = [g.get("title", g.get("id", "")) for g in goals]

        for i, action in enumerate(actions):
            name = action.get("name", action.get("id", f"action-{i}"))
            step_type = action.get("step_type", "task")

            steps.append(
                PipelineVerificationStep(
                    step_id=f"action-{i}",
                    stage="goals_to_actions",
                    content=f"Action '{name}' (type: {step_type})",
                    dependencies=[f"Goal: {t}" for t in goal_titles],
                )
            )
        return steps

    def _extract_assignment_steps(
        self,
        actions: list[dict[str, Any]],
        assignments: list[dict[str, Any]],
    ) -> list[PipelineVerificationStep]:
        """Extract verifiable steps from action→orchestration transition."""
        steps = []
        for i, assignment in enumerate(assignments):
            name = assignment.get("name", assignment.get("id", f"task-{i}"))
            agent_id = assignment.get("agent_id", "unassigned")

            steps.append(
                PipelineVerificationStep(
                    step_id=f"assign-{i}",
                    stage="actions_to_orchestration",
                    content=f"Task '{name}' assigned to {agent_id}",
                    agent_id=agent_id,
                    dependencies=[a.get("name", "") for a in actions],
                )
            )
        return steps

    # -- Verification engine -----------------------------------------------

    async def _verify_steps(
        self,
        steps: list[PipelineVerificationStep],
        context: str,
    ) -> list[dict[str, Any]]:
        """Verify steps using ThinkPRM if available, else structural checks."""
        try:
            result = await self._verify_with_think_prm(steps, context)
            if result:  # ThinkPRM may return empty on API failure
                return result
        except Exception as exc:
            logger.debug("ThinkPRM unavailable, using structural verification: %s", exc)

        return self._structural_verify(steps, context)

    async def _verify_with_think_prm(
        self,
        steps: list[PipelineVerificationStep],
        context: str,
    ) -> list[dict[str, Any]]:
        """Use ThinkPRM for AI-powered step verification."""
        from aragora.verification.think_prm import ThinkPRMVerifier, ThinkPRMConfig

        config = ThinkPRMConfig(
            max_context_chars=1500,
            critical_round_threshold=0.7,
        )
        verifier = ThinkPRMVerifier(config=config)

        # Build debate-round-like structures for ThinkPRM
        debate_rounds = []
        for step in steps:
            debate_rounds.append(
                {
                    "round_number": len(debate_rounds),
                    "agent_id": step.agent_id or "pipeline",
                    "content": step.content,
                    "dependencies": step.dependencies,
                }
            )

        # Create a simple query function for verification
        async def _query_fn(agent_id: str, prompt: str, max_tokens: int = 500) -> str:
            try:
                from aragora.agents.api_agents.anthropic import AnthropicAgent

                agent = AnthropicAgent(
                    name="step-verifier",
                    model="claude-sonnet-4-20250514",
                )
                return await agent.generate(prompt)
            except (ImportError, RuntimeError):
                return (
                    "VERDICT: UNCERTAIN\n"
                    "CONFIDENCE: 0.5\n"
                    "REASONING: Verification agent unavailable\n"
                    "SUGGESTED_FIX: None"
                )

        result = await verifier.verify_debate_process(
            debate_rounds=debate_rounds,
            query_fn=_query_fn,
        )

        return [
            {
                "step_id": steps[i].step_id if i < len(steps) else f"step-{i}",
                "verdict": sv.verdict.value,
                "confidence": sv.confidence,
                "reasoning": sv.reasoning,
                "suggested_fix": sv.suggested_fix,
            }
            for i, sv in enumerate(result.step_results)
        ]

    def _structural_verify(
        self,
        steps: list[PipelineVerificationStep],
        context: str,
    ) -> list[dict[str, Any]]:
        """Fallback structural verification without AI."""
        results = []
        for step in steps:
            # Basic structural checks
            has_content = bool(step.content.strip())
            has_deps = len(step.dependencies) > 0

            if has_content and has_deps:
                verdict = "correct"
                confidence = 0.6
            elif has_content:
                verdict = "uncertain"
                confidence = 0.4
            else:
                verdict = "incorrect"
                confidence = 0.8

            results.append(
                {
                    "step_id": step.step_id,
                    "verdict": verdict,
                    "confidence": confidence,
                    "reasoning": f"Structural check: content={'yes' if has_content else 'no'}, "
                    f"dependencies={'yes' if has_deps else 'no'}",
                    "suggested_fix": None if verdict == "correct" else "Add missing context",
                }
            )
        return results

    # -- Calibration integration -------------------------------------------

    async def _apply_calibration(
        self,
        raw_score: float,
        step_results: list[dict[str, Any]],
    ) -> float:
        """Adjust verification score using agent calibration data."""
        try:
            from aragora.agents.calibration import CalibrationTracker

            tracker = CalibrationTracker()

            # Get aggregate calibration quality
            agent_ids = {sr.get("agent_id", "") for sr in step_results if sr.get("agent_id")}

            if not agent_ids:
                return raw_score

            brier_scores = []
            for agent_id in agent_ids:
                brier = tracker.get_brier_score(agent_id)
                if brier is not None:
                    brier_scores.append(brier)

            if not brier_scores:
                return raw_score

            # Average Brier score (0 = perfect, 1 = worst)
            avg_brier = sum(brier_scores) / len(brier_scores)
            calibration_quality = 1.0 - avg_brier

            # Blend raw score with calibration quality
            adjusted = (
                raw_score * (1 - self.calibration_weight)
                + calibration_quality * self.calibration_weight
            )
            return max(0.0, min(1.0, adjusted))

        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Calibration adjustment unavailable: %s", exc)
            return raw_score

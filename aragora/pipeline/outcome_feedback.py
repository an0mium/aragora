"""Pipeline Outcome Feedback System.

Records structured learning from every pipeline run and bridges
results to KnowledgeMound, ELO system, and Interrogation Calibrator
for continuous improvement.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentContribution:
    """Record of an agent's contribution to a pipeline run."""

    agent_name: str
    provider: str
    phase: str  # debate, interrogation, execution, verification
    influence_score: float = 0.0  # 0-1, how much this agent shaped outcome
    truth_ratio: float = 0.0  # from TruthScorer
    calibration_error: float = 0.0  # |predicted - actual| confidence


@dataclass
class PipelineOutcome:
    """Comprehensive outcome record from a pipeline run."""

    pipeline_id: str
    run_type: str  # self_improvement, user_project, debate_only
    domain: str

    # Interrogation phase
    questions_asked: int = 0
    questions_answered: int = 0
    answers_changed_default: int = 0

    # Spec quality
    spec_completeness: float = 0.0  # 0-1

    # Execution
    execution_succeeded: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    files_changed: int = 0
    rollback_triggered: bool = False

    # Human interaction
    human_interventions: int = 0

    # Timing
    total_duration_s: float = 0.0

    # Agent contributions
    agent_contributions: list[AgentContribution] = field(default_factory=list)

    @property
    def overall_quality_score(self) -> float:
        """Compute composite quality score (0-1).

        Weights:
        - 40% execution success + test results
        - 20% spec completeness
        - 15% interrogation effectiveness
        - 15% human intervention (fewer = better)
        - 10% no rollback
        """
        # Execution: succeeded + test pass ratio
        if self.tests_passed + self.tests_failed > 0:
            test_ratio = self.tests_passed / (self.tests_passed + self.tests_failed)
        else:
            test_ratio = 1.0 if self.execution_succeeded else 0.0

        exec_score = (0.6 * (1.0 if self.execution_succeeded else 0.0)) + (0.4 * test_ratio)

        # Interrogation effectiveness
        if self.questions_asked > 0:
            interrog_score = self.questions_answered / self.questions_asked
        else:
            interrog_score = 0.5  # Neutral when no questions

        # Human intervention (fewer is better, cap at 5)
        human_score = max(0.0, 1.0 - (self.human_interventions / 5.0))

        # No rollback
        rollback_score = 0.0 if self.rollback_triggered else 1.0

        return (
            0.40 * exec_score
            + 0.20 * self.spec_completeness
            + 0.15 * interrog_score
            + 0.15 * human_score
            + 0.10 * rollback_score
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for KM ingestion."""
        return {
            "type": "pipeline_outcome",
            "pipeline_id": self.pipeline_id,
            "run_type": self.run_type,
            "domain": self.domain,
            "quality_score": round(self.overall_quality_score, 3),
            "interrogation": {
                "questions_asked": self.questions_asked,
                "questions_answered": self.questions_answered,
                "answers_changed_default": self.answers_changed_default,
            },
            "execution": {
                "succeeded": self.execution_succeeded,
                "tests_passed": self.tests_passed,
                "tests_failed": self.tests_failed,
                "files_changed": self.files_changed,
                "rollback_triggered": self.rollback_triggered,
            },
            "spec_completeness": self.spec_completeness,
            "human_interventions": self.human_interventions,
            "total_duration_s": self.total_duration_s,
            "agents": {
                "contributions": [
                    {
                        "agent_name": c.agent_name,
                        "provider": c.provider,
                        "phase": c.phase,
                        "influence_score": c.influence_score,
                        "truth_ratio": c.truth_ratio,
                        "calibration_error": c.calibration_error,
                    }
                    for c in self.agent_contributions
                ]
            },
        }


class OutcomeFeedbackRecorder:
    """Records pipeline outcomes and bridges to learning systems.

    Integrates with:
    - KnowledgeMound: persists outcomes for cross-cycle learning
    - ELO system: updates agent ratings per domain:phase
    - Calibrator: feeds interrogation effectiveness data
    """

    def __init__(
        self,
        knowledge_mound: Any | None = None,
        elo_system: Any | None = None,
        calibrator: Any | None = None,
    ) -> None:
        self._outcomes: list[PipelineOutcome] = []
        self._km = knowledge_mound
        self._elo = elo_system
        self._calibrator = calibrator

    def record(self, outcome: PipelineOutcome) -> None:
        """Record an outcome and propagate to learning systems."""
        self._outcomes.append(outcome)

        # Bridge to KnowledgeMound
        if self._km is not None:
            try:
                self._km.ingest(outcome.to_dict())
            except Exception:
                logger.warning("Failed to ingest outcome to KM: %s", outcome.pipeline_id)

        # Bridge to ELO
        if self._elo is not None:
            for contrib in outcome.agent_contributions:
                domain_phase = f"{outcome.domain}:{contrib.phase}"
                won = contrib.influence_score >= 0.5
                self._elo.update_domain_elo(contrib.agent_name, domain_phase, won=won)

        # Bridge to calibrator
        if self._calibrator is not None:
            try:
                self._calibrator.record_pipeline_outcome(outcome)
            except Exception:
                logger.warning("Failed to update calibrator: %s", outcome.pipeline_id)

    def get_recent_outcomes(
        self,
        limit: int = 10,
        run_type: str | None = None,
    ) -> list[PipelineOutcome]:
        """Get recent outcomes, optionally filtered by run type."""
        outcomes = self._outcomes
        if run_type is not None:
            outcomes = [o for o in outcomes if o.run_type == run_type]
        return outcomes[-limit:]

    def get_quality_trend(self, window: int = 10) -> list[float]:
        """Get quality score trend over recent outcomes."""
        recent = self._outcomes[-window:]
        return [o.overall_quality_score for o in recent]

    def get_agent_phase_performance(
        self,
    ) -> dict[str, dict[str, float]]:
        """Get average influence score per agent per phase.

        Returns:
            {agent_name: {phase: avg_influence_score}}
        """
        scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for outcome in self._outcomes:
            for contrib in outcome.agent_contributions:
                scores[contrib.agent_name][contrib.phase].append(contrib.influence_score)

        result: dict[str, dict[str, float]] = {}
        for agent, phases in scores.items():
            result[agent] = {phase: sum(vals) / len(vals) for phase, vals in phases.items()}
        return result

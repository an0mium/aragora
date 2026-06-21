"""Execution-pattern selection for the hybrid orchestrator.

Picks *how* a unit of work should be executed -- which of three orchestration
patterns best fits -- distinct from aragora's operational :class:`~aragora.modes.base.Mode`
(Architect/Coder/RedTeam/...), which shapes *what* an agent may do.

The three patterns map to existing executors:

* ``DYNAMIC_WORKFLOW`` -> ``aragora/workflow/engine.py`` -- adaptive, decomposable,
  deterministic multi-step work; the lowest-overhead default.
* ``GOAL_ANCHORED``    -> ``aragora/nomic/autonomous_orchestrator.py`` -- abstract,
  open-ended goals ("improve X") needing decomposition + parallel tracks.
* ``AGENT_TEAMS``      -> ``aragora/debate/orchestrator.py`` (Arena) -- decisions
  needing adversarial consensus; reserved for risk/consensus/contested work
  because it is the most expensive executor.

This module is **pure and deterministic** -- no I/O, no model calls -- so it is
unit-testable and safe to ship unwired. ``goal_abstractness`` and
``complexity_score`` are inputs the caller supplies (e.g. from ``DomainDetector``);
``estimate_goal_abstractness`` offers a keyword fallback. Per the project's "use
real intelligence, not regex" principle the heuristic is the *fallback*: an
opt-in LLM-as-judge path is the intended steady state (a follow-on), and the
heuristic exists so the feature ships and degrades safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OrchestrationPattern(Enum):
    """How a unit of work is executed. (Distinct from ``workflow.types.ExecutionPattern``.)"""

    DYNAMIC_WORKFLOW = "dynamic_workflow"
    GOAL_ANCHORED = "goal_anchored"
    AGENT_TEAMS = "agent_teams"


# Informational map pattern -> executor module (no import; documentation/telemetry only).
EXECUTOR_MODULES: dict[OrchestrationPattern, str] = {
    OrchestrationPattern.DYNAMIC_WORKFLOW: "aragora.workflow.engine",
    OrchestrationPattern.GOAL_ANCHORED: "aragora.nomic.autonomous_orchestrator",
    OrchestrationPattern.AGENT_TEAMS: "aragora.debate.orchestrator",
}

_ABSTRACT_GOAL_MARKERS = (
    "maximize",
    "minimize",
    "improve",
    "optimize",
    "enhance",
    "make better",
    "increase",
    "reduce",
)


def estimate_goal_abstractness(task_text: str) -> float:
    """Heuristic 0-1 abstractness for a task description (fallback only).

    Open-ended optimization verbs without a concrete deliverable read as
    abstract; an imperative with a concrete object reads as concrete. This is a
    deliberate keyword fallback -- the intended steady state is an LLM judge.
    """
    text = (task_text or "").strip().lower()
    if not text:
        return 0.0
    return 0.7 if any(marker in text for marker in _ABSTRACT_GOAL_MARKERS) else 0.0


@dataclass(frozen=True)
class ModeDecisionContext:
    """Inputs to pattern selection. All optional with safe defaults for testing.

    ``risk_tier`` 0-4 (PR/goal classification). ``complexity_score`` 0-10.
    ``goal_abstractness`` 0-1. The booleans are cheap structural signals the
    caller already has from intake/classification.
    """

    task_text: str = ""
    risk_tier: int = 0
    consensus_required: bool = False
    complexity_score: float = 0.0
    goal_abstractness: float = 0.0
    is_code_change: bool = False
    is_design: bool = False
    involves_error: bool = False
    domain: str = "general"
    prior_pattern: OrchestrationPattern | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.risk_tier <= 4:
            raise ValueError("risk_tier must be in [0, 4]")
        if not 0.0 <= self.complexity_score <= 10.0:
            raise ValueError("complexity_score must be in [0, 10]")
        if not 0.0 <= self.goal_abstractness <= 1.0:
            raise ValueError("goal_abstractness must be in [0, 1]")


@dataclass(frozen=True)
class PatternDecision:
    """Result of pattern selection."""

    pattern: OrchestrationPattern
    confidence: float
    rationale: str

    @property
    def executor_module(self) -> str:
        return EXECUTOR_MODULES[self.pattern]

    def to_dict(self) -> dict[str, object]:
        return {
            "pattern": self.pattern.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "executor_module": self.executor_module,
        }


class OperationalModeSelector:
    """Deterministic execution-pattern selector.

    Ordering reserves the expensive ``AGENT_TEAMS`` pattern for work that
    genuinely needs adversarial vetting (risk/consensus/contested/design/error/
    very-complex) and defaults everything else to the lowest-overhead
    ``DYNAMIC_WORKFLOW`` -- so an unclassified medium task never silently lands
    on the most expensive executor.
    """

    def select_pattern(self, ctx: ModeDecisionContext) -> PatternDecision:
        # 1. Hard safety override: high-risk or explicitly contested -> teams.
        if ctx.risk_tier >= 3 or ctx.consensus_required:
            return PatternDecision(
                OrchestrationPattern.AGENT_TEAMS, 0.95, "risk/consensus gate -> adversarial teams"
            )

        # 2. Genuine decisions / design / error-diagnosis / very complex -> teams.
        if ctx.is_design or ctx.involves_error or ctx.complexity_score >= 8.0:
            return PatternDecision(
                OrchestrationPattern.AGENT_TEAMS,
                0.8,
                "contested correctness -> adversarial vetting",
            )

        # 3. Abstract, open-ended, non-code goals -> goal-anchored decomposition.
        if ctx.goal_abstractness >= 0.6 and not ctx.is_code_change:
            return PatternDecision(
                OrchestrationPattern.GOAL_ANCHORED, 0.8, "abstract goal needs decomposition"
            )

        # 4. Structured, decomposable, deterministic multi-step -> dynamic workflow.
        if 3.0 <= ctx.complexity_score <= 7.0:
            return PatternDecision(
                OrchestrationPattern.DYNAMIC_WORKFLOW, 0.7, "structured multi-step task"
            )

        # 5. Default: lowest-overhead pattern (never default to the costly teams path).
        #    A prior successful pattern (KM hint) breaks ties when present.
        if ctx.prior_pattern is not None:
            return PatternDecision(ctx.prior_pattern, 0.55, "default with prior-outcome tiebreaker")
        return PatternDecision(
            OrchestrationPattern.DYNAMIC_WORKFLOW, 0.6, "default: lowest-overhead pattern"
        )

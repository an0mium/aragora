"""Scores the GTI corpus under naive and gated policies."""

from __future__ import annotations

from dataclasses import dataclass

from aragora.gti.policies import PolicyOutcome, gated_policy, naive_policy
from aragora.gti.scenarios import Scenario


@dataclass(frozen=True)
class Metrics:
    stale_belief_action_rate: float
    detection_rate: float
    correction_rate: float
    false_green_rate: float


@dataclass(frozen=True)
class ScoreResult:
    naive: Metrics
    gated: Metrics
    delta: Metrics  # naive - gated (positive = gated improved)
    scenario_count: int


def _rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def _metrics(scenarios: list[Scenario], outcomes: list[PolicyOutcome]) -> Metrics:
    total = len(scenarios)
    wrong_idx = [i for i, s in enumerate(scenarios) if not s.belief_matches_truth]
    detected = sum(1 for i in wrong_idx if outcomes[i].detected_stale)
    corrected = sum(1 for i in wrong_idx if outcomes[i].corrected)
    return Metrics(
        stale_belief_action_rate=_rate(sum(o.acted_on_stale_belief for o in outcomes), total),
        detection_rate=_rate(detected, len(wrong_idx)),
        correction_rate=_rate(corrected, len(wrong_idx)),
        false_green_rate=_rate(sum(o.reported_green_but_wrong for o in outcomes), total),
    )


def score_corpus(scenarios: list[Scenario]) -> ScoreResult:
    naive_out = [naive_policy(s) for s in scenarios]
    gated_out = [gated_policy(s) for s in scenarios]
    naive = _metrics(scenarios, naive_out)
    gated = _metrics(scenarios, gated_out)
    delta = Metrics(
        stale_belief_action_rate=naive.stale_belief_action_rate - gated.stale_belief_action_rate,
        detection_rate=gated.detection_rate - naive.detection_rate,
        correction_rate=gated.correction_rate - naive.correction_rate,
        false_green_rate=naive.false_green_rate - gated.false_green_rate,
    )
    return ScoreResult(naive=naive, gated=gated, delta=delta, scenario_count=len(scenarios))

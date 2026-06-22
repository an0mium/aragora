"""Deterministic naive vs gated decision policies over GTI scenarios.

These model the *gating logic* (freshness gate + canonical re-derivation +
heterogeneous-quorum disagreement), not live LLM calls, so the benchmark is
deterministic and independent of agent availability. Swapping in real agents
is a future iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

from aragora.gti.scenarios import Scenario


@dataclass(frozen=True)
class PolicyOutcome:
    acted_on_stale_belief: bool
    detected_stale: bool
    corrected: bool
    reported_green_but_wrong: bool


def naive_policy(scenario: Scenario) -> PolicyOutcome:
    """Acts on the presented belief with no freshness/canonical/quorum check."""
    wrong = not scenario.belief_matches_truth
    return PolicyOutcome(
        acted_on_stale_belief=wrong,
        detected_stale=False,
        corrected=False,
        reported_green_but_wrong=wrong,
    )


def gated_policy(scenario: Scenario) -> PolicyOutcome:
    """Applies the freshness gate (age>TTL) and quorum-disagreement signal.

    When either fires, the belief is re-derived from the canonical source and
    corrected. When neither fires, the gate behaves like naive (an honest miss
    if the belief was wrong but undetectable).
    """
    stale_by_age = scenario.belief_age_days > scenario.freshness_ttl_days
    flagged = stale_by_age or scenario.quorum_would_flag
    if flagged:
        return PolicyOutcome(
            acted_on_stale_belief=False,
            detected_stale=True,
            corrected=not scenario.belief_matches_truth,
            reported_green_but_wrong=False,
        )
    wrong = not scenario.belief_matches_truth
    return PolicyOutcome(
        acted_on_stale_belief=wrong,
        detected_stale=False,
        corrected=False,
        reported_green_but_wrong=wrong,
    )

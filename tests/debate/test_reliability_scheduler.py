"""Tests for reliability-aware budget scheduler."""

from __future__ import annotations

from aragora.debate.epistemic_outcomes import EpistemicOutcome
from aragora.debate.reliability_scheduler import ReliabilityScheduler


def test_score_from_calibration_in_expected_range():
    scheduler = ReliabilityScheduler()
    score = scheduler.score_from_calibration(
        {
            "brier_score": 0.12,
            "ece": 0.06,
            "prediction_count": 40,
        }
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.7


def test_build_settlement_deltas_aggregates_by_participant():
    scheduler = ReliabilityScheduler()
    outcomes = [
        EpistemicOutcome(
            debate_id="d1",
            claim="c1",
            falsifier="f1",
            metric="m1",
            status="resolved",
            confidence_delta=0.2,
            metadata={"participants": ["claude", "gpt4"]},
        ),
        EpistemicOutcome(
            debate_id="d2",
            claim="c2",
            falsifier="f2",
            metric="m2",
            status="resolved",
            confidence_delta=-0.1,
            metadata={"participants": ["gpt4"]},
        ),
        EpistemicOutcome(
            debate_id="d3",
            claim="c3",
            falsifier="f3",
            metric="m3",
            status="open",
            confidence_delta=1.0,
            metadata={"participants": ["claude"]},
        ),
    ]
    deltas = scheduler.build_settlement_deltas(outcomes)
    assert deltas["claude"] == 0.2
    assert deltas["gpt4"] == 0.1


def test_allocate_budget_is_normalized_and_respects_floor():
    scheduler = ReliabilityScheduler(min_share=0.1)
    shares = scheduler.allocate_budget(
        ["claude", "gpt4", "gemini"],
        calibration_map={
            "claude": {"brier_score": 0.1, "ece": 0.05, "prediction_count": 50},
            "gpt4": {"brier_score": 0.2, "ece": 0.08, "prediction_count": 35},
            "gemini": {"brier_score": 0.45, "ece": 0.25, "prediction_count": 5},
        },
        settlement_deltas={"claude": 0.3, "gpt4": 0.05, "gemini": -0.2},
    )

    assert set(shares.keys()) == {"claude", "gpt4", "gemini"}
    assert abs(sum(shares.values()) - 1.0) < 1e-6
    assert all(v >= 0.1 - 1e-6 for v in shares.values())
    assert shares["claude"] > shares["gemini"]

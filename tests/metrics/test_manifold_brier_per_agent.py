"""Tests for per-agent Brier scoring (AGT-03 sub-deliverable 3/4).

Covers ManifoldBrierScorer.agents(), .rolling_score_for_agent(),
and .calibration_curve_for_agent().
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aragora.metrics.manifold_brier import (
    ManifoldBrierScorer,
    ManifoldPrediction,
)


def _pred(
    p: float,
    outcome: int,
    *,
    agent_id: str = "alpha",
    days_ago: float = 0,
    question_id: str | None = None,
) -> ManifoldPrediction:
    return ManifoldPrediction(
        question_id=question_id or f"q_{agent_id}_{p}_{days_ago}",
        predicted_probability=p,
        outcome=outcome,
        predicted_at=datetime.now(UTC) - timedelta(days=days_ago),
        agent_id=agent_id,
    )


@pytest.fixture(autouse=True)
def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")


# ---------------------------------------------------------------------------
# Feature-gate enforcement
# ---------------------------------------------------------------------------


class TestGate:
    @pytest.mark.parametrize(
        "method,args",
        [
            ("agents", []),
            ("rolling_score_for_agent", ["x"]),
            ("calibration_curve_for_agent", ["x"]),
        ],
    )
    def test_raises_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch, method: str, args: list
    ) -> None:
        monkeypatch.delenv("ARAGORA_MANIFOLD_BRIER_ENABLED", raising=False)
        scorer = ManifoldBrierScorer()
        with pytest.raises(RuntimeError, match="disabled"):
            getattr(scorer, method)(*args)


# ---------------------------------------------------------------------------
# agents()
# ---------------------------------------------------------------------------


class TestAgents:
    def test_empty_scorer(self) -> None:
        assert ManifoldBrierScorer().agents() == []

    def test_sorted_and_deduplicated(self) -> None:
        scorer = ManifoldBrierScorer()
        for q in "abc":
            scorer.add(_pred(0.5, 1, agent_id="zebra", question_id=f"z{q}"))
        scorer.add(_pred(0.5, 1, agent_id="alpha", question_id="a1"))
        assert scorer.agents() == ["alpha", "zebra"]

    def test_blank_agent_id_excluded(self) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(
            ManifoldPrediction(
                question_id="anon",
                predicted_probability=0.5,
                outcome=1,
                predicted_at=datetime.now(UTC),
                agent_id="",
            )
        )
        scorer.add(_pred(0.7, 1, agent_id="alpha"))
        assert scorer.agents() == ["alpha"]


# ---------------------------------------------------------------------------
# rolling_score_for_agent()
# ---------------------------------------------------------------------------


class TestRollingScoreForAgent:
    def test_empty_returns_none_stats(self) -> None:
        s = ManifoldBrierScorer()
        r = s.rolling_score_for_agent("alpha")
        assert r.n_predictions == 0
        assert r.mean_brier is None

    def test_default_window_90_days(self) -> None:
        s = ManifoldBrierScorer()
        s.add(_pred(0.8, 1, days_ago=89, question_id="in"))
        s.add(_pred(0.8, 1, days_ago=91, question_id="out"))
        r = s.rolling_score_for_agent("alpha")
        assert r.window_days == 90
        assert r.n_predictions == 1

    def test_excludes_other_agents(self) -> None:
        s = ManifoldBrierScorer()
        s.add(_pred(1.0, 1, agent_id="alpha", question_id="a"))
        s.add(_pred(0.0, 1, agent_id="beta", question_id="b"))
        assert s.rolling_score_for_agent("alpha").n_predictions == 1
        assert s.rolling_score_for_agent("beta").n_predictions == 1

    def test_two_agents_independent_scores(self) -> None:
        s = ManifoldBrierScorer()
        s.add(_pred(1.0, 1, agent_id="alpha", question_id="a"))  # BS=0.0
        s.add(_pred(0.0, 1, agent_id="beta", question_id="b"))  # BS=1.0
        assert s.rolling_score_for_agent("alpha").mean_brier == pytest.approx(0.0)
        assert s.rolling_score_for_agent("beta").mean_brier == pytest.approx(1.0)

    def test_empty_agent_id_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            ManifoldBrierScorer().rolling_score_for_agent("")

    def test_invalid_window_days_raises(self) -> None:
        with pytest.raises(ValueError, match="window_days"):
            ManifoldBrierScorer().rolling_score_for_agent("alpha", window_days=0)

    def test_unknown_agent_returns_empty(self) -> None:
        s = ManifoldBrierScorer()
        s.add(_pred(0.7, 1, agent_id="alpha"))
        assert s.rolling_score_for_agent("no-such-agent").n_predictions == 0


# ---------------------------------------------------------------------------
# calibration_curve_for_agent()
# ---------------------------------------------------------------------------


class TestCalibrationCurveForAgent:
    def test_returns_n_bins(self) -> None:
        assert len(ManifoldBrierScorer().calibration_curve_for_agent("x", n_bins=5)) == 5

    def test_empty_scorer_all_empty_bins(self) -> None:
        curve = ManifoldBrierScorer().calibration_curve_for_agent("x")
        assert all(b.count == 0 and b.fraction_yes is None for b in curve)

    def test_other_agent_excluded(self) -> None:
        s = ManifoldBrierScorer()
        s.add(_pred(0.75, 1, agent_id="beta"))
        assert sum(b.count for b in s.calibration_curve_for_agent("alpha")) == 0

    def test_fraction_yes_correct(self) -> None:
        s = ManifoldBrierScorer()
        s.add(_pred(0.25, 1, agent_id="x", question_id="y1"))
        s.add(_pred(0.25, 0, agent_id="x", question_id="n1"))
        curve = s.calibration_curve_for_agent("x", n_bins=10)
        assert curve[2].count == 2
        assert curve[2].fraction_yes == pytest.approx(0.5)

    def test_p1_clamps_to_last_bin(self) -> None:
        s = ManifoldBrierScorer()
        s.add(_pred(1.0, 1, agent_id="x"))
        curve = s.calibration_curve_for_agent("x", n_bins=10)
        assert curve[-1].count == 1
        assert sum(b.count for b in curve) == 1

    def test_empty_agent_id_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            ManifoldBrierScorer().calibration_curve_for_agent("")

    def test_invalid_n_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            ManifoldBrierScorer().calibration_curve_for_agent("x", n_bins=1)

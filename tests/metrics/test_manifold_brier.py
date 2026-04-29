"""Tests for aragora.metrics.manifold_brier — Manifold Brier scorer skeleton."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from aragora.metrics.manifold_brier import (
    BrierWindowSummary,
    CalibrationBin,
    ManifoldBrierScorer,
    ManifoldPrediction,
    brier_score,
    manifold_brier_enabled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(
    p: float,
    outcome: int,
    days_ago: float = 0,
    question_id: str = "q1",
) -> ManifoldPrediction:
    now = datetime.now(UTC)
    return ManifoldPrediction(
        question_id=question_id,
        predicted_probability=p,
        outcome=outcome,
        predicted_at=now - timedelta(days=days_ago),
    )


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------


class TestFeatureGate:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_MANIFOLD_BRIER_ENABLED", raising=False)
        assert manifold_brier_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "YES"])
    def test_enabled_truthy_values(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", val)
        assert manifold_brier_enabled() is True

    def test_disabled_falsy_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "0")
        assert manifold_brier_enabled() is False

    def test_scorer_raises_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_MANIFOLD_BRIER_ENABLED", raising=False)
        scorer = ManifoldBrierScorer()
        with pytest.raises(RuntimeError, match="disabled"):
            scorer.add(_pred(0.7, 1))


# ---------------------------------------------------------------------------
# brier_score pure function
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_yes_prediction(self) -> None:
        assert brier_score(1.0, 1) == pytest.approx(0.0)

    def test_perfect_no_prediction(self) -> None:
        assert brier_score(0.0, 0) == pytest.approx(0.0)

    def test_worst_yes_prediction(self) -> None:
        assert brier_score(0.0, 1) == pytest.approx(1.0)

    def test_worst_no_prediction(self) -> None:
        assert brier_score(1.0, 0) == pytest.approx(1.0)

    def test_uninformative_prediction(self) -> None:
        assert brier_score(0.5, 1) == pytest.approx(0.25)
        assert brier_score(0.5, 0) == pytest.approx(0.25)

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="predicted_probability"):
            brier_score(1.1, 1)
        with pytest.raises(ValueError, match="predicted_probability"):
            brier_score(-0.1, 0)

    def test_invalid_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="outcome"):
            brier_score(0.5, 2)


# ---------------------------------------------------------------------------
# ManifoldPrediction
# ---------------------------------------------------------------------------


class TestManifoldPrediction:
    def test_score_property(self) -> None:
        p = ManifoldPrediction(
            question_id="q1",
            predicted_probability=0.8,
            outcome=1,
            predicted_at=datetime.now(UTC),
        )
        assert p.score == pytest.approx(0.04)

    def test_to_dict_shape(self) -> None:
        p = _pred(0.6, 0)
        d = p.to_dict()
        assert "brier_score" in d
        assert "predicted_probability" in d
        assert "outcome" in d

    def test_invalid_probability_raises_on_construction(self) -> None:
        with pytest.raises(ValueError):
            ManifoldPrediction(
                question_id="q",
                predicted_probability=1.5,
                outcome=1,
                predicted_at=datetime.now(UTC),
            )


# ---------------------------------------------------------------------------
# ManifoldBrierScorer
# ---------------------------------------------------------------------------


class TestManifoldBrierScorer:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")

    def test_empty_window_returns_none_stats(self) -> None:
        scorer = ManifoldBrierScorer()
        summary = scorer.rolling_score(window_days=30)
        assert summary.n_predictions == 0
        assert summary.mean_brier is None
        assert summary.median_brier is None

    def test_single_prediction_in_window(self) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(1.0, 1, days_ago=0))
        summary = scorer.rolling_score(window_days=7)
        assert summary.n_predictions == 1
        assert summary.mean_brier == pytest.approx(0.0)

    def test_prediction_outside_window_excluded(self) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(0.9, 1, days_ago=60))
        summary = scorer.rolling_score(window_days=30)
        assert summary.n_predictions == 0

    def test_mean_brier_across_multiple_predictions(self) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(1.0, 1, days_ago=1, question_id="a"))  # BS=0.0
        scorer.add(_pred(0.0, 0, days_ago=2, question_id="b"))  # BS=0.0
        scorer.add(_pred(0.5, 1, days_ago=3, question_id="c"))  # BS=0.25
        summary = scorer.rolling_score(window_days=30)
        assert summary.n_predictions == 3
        assert summary.mean_brier == pytest.approx(0.25 / 3)

    def test_invalid_window_days_raises(self) -> None:
        scorer = ManifoldBrierScorer()
        with pytest.raises(ValueError, match="window_days"):
            scorer.rolling_score(window_days=0)

    def test_clear_removes_all_predictions(self) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(0.9, 1))
        scorer.clear()
        assert scorer.rolling_score(window_days=7).n_predictions == 0

    def test_to_dict_on_window_summary(self) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(0.7, 1))
        summary = scorer.rolling_score(window_days=30)
        d = summary.to_dict()
        assert "window_days" in d
        assert "mean_brier" in d
        assert "n_predictions" in d


# ---------------------------------------------------------------------------
# CalibrationBin
# ---------------------------------------------------------------------------


class TestCalibrationBin:
    def test_to_dict_shape(self) -> None:
        b = CalibrationBin(low=0.1, high=0.2, count=3, fraction_yes=0.67, mean_predicted=0.15)
        d = b.to_dict()
        assert d == {
            "low": 0.1,
            "high": 0.2,
            "count": 3,
            "fraction_yes": 0.67,
            "mean_predicted": 0.15,
        }

    def test_empty_bin_nones(self) -> None:
        b = CalibrationBin(low=0.0, high=0.1, count=0, fraction_yes=None, mean_predicted=None)
        assert b.fraction_yes is None
        assert b.mean_predicted is None


# ---------------------------------------------------------------------------
# ManifoldBrierScorer.calibration_curve
# ---------------------------------------------------------------------------


class TestCalibrationCurve:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")

    def test_returns_n_bins_entries(self) -> None:
        scorer = ManifoldBrierScorer()
        curve = scorer.calibration_curve(n_bins=10)
        assert len(curve) == 10

    def test_empty_scorer_all_empty_bins(self) -> None:
        scorer = ManifoldBrierScorer()
        curve = scorer.calibration_curve(n_bins=5)
        assert all(b.count == 0 for b in curve)
        assert all(b.fraction_yes is None for b in curve)
        assert all(b.mean_predicted is None for b in curve)

    def test_bin_boundaries_are_contiguous(self) -> None:
        scorer = ManifoldBrierScorer()
        curve = scorer.calibration_curve(n_bins=4)
        for i in range(len(curve) - 1):
            assert curve[i].high == pytest.approx(curve[i + 1].low)

    def test_first_bin_low_is_zero(self) -> None:
        scorer = ManifoldBrierScorer()
        curve = scorer.calibration_curve(n_bins=10)
        assert curve[0].low == pytest.approx(0.0)

    def test_last_bin_high_is_one(self) -> None:
        scorer = ManifoldBrierScorer()
        curve = scorer.calibration_curve(n_bins=10)
        assert curve[-1].high == pytest.approx(1.0)

    def test_probability_one_falls_in_last_bin(self) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(1.0, 1))
        curve = scorer.calibration_curve(n_bins=10)
        # p=1.0 must land in the last bin, not overflow
        assert curve[-1].count == 1
        total = sum(b.count for b in curve)
        assert total == 1

    def test_fraction_yes_perfect_calibration(self) -> None:
        # All predictions at ~0.75 → last bin before 0.8 should have fraction_yes=1.0
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(0.75, 1, question_id="a"))
        scorer.add(_pred(0.75, 1, question_id="b"))
        curve = scorer.calibration_curve(n_bins=10)
        # p=0.75 → bin index 7 ([0.7, 0.8))
        bin_7 = curve[7]
        assert bin_7.count == 2
        assert bin_7.fraction_yes == pytest.approx(1.0)
        assert bin_7.mean_predicted == pytest.approx(0.75)

    def test_fraction_yes_mixed_outcomes(self) -> None:
        scorer = ManifoldBrierScorer()
        # Put 2 YES and 2 NO in the 0.2–0.3 bracket (p=0.25)
        for _ in range(2):
            scorer.add(_pred(0.25, 1, question_id=f"yes_{_}"))
        for _ in range(2):
            scorer.add(_pred(0.25, 0, question_id=f"no_{_}"))
        curve = scorer.calibration_curve(n_bins=10)
        # p=0.25 → bin index 2 ([0.2, 0.3))
        bin_2 = curve[2]
        assert bin_2.count == 4
        assert bin_2.fraction_yes == pytest.approx(0.5)

    def test_total_count_equals_num_predictions(self) -> None:
        scorer = ManifoldBrierScorer()
        for i in range(6):
            scorer.add(_pred(i / 10, 0, question_id=f"q{i}"))
        curve = scorer.calibration_curve(n_bins=10)
        assert sum(b.count for b in curve) == 6

    def test_invalid_n_bins_raises(self) -> None:
        scorer = ManifoldBrierScorer()
        with pytest.raises(ValueError, match="n_bins"):
            scorer.calibration_curve(n_bins=1)
        with pytest.raises(ValueError, match="n_bins"):
            scorer.calibration_curve(n_bins=101)

    def test_disabled_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_MANIFOLD_BRIER_ENABLED", raising=False)
        scorer = ManifoldBrierScorer()
        with pytest.raises(RuntimeError, match="disabled"):
            scorer.calibration_curve()

    def test_custom_n_bins(self) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(0.5, 1))
        curve = scorer.calibration_curve(n_bins=20)
        assert len(curve) == 20
        assert sum(b.count for b in curve) == 1

    @pytest.mark.parametrize(
        "prob,expected_bin",
        [
            # Exact decimal boundaries that are slightly below their target
            # value in IEEE 754, triggering int(p / step) = n-1 instead of n.
            (0.3, 3),  # 0.3/0.1 == 2.9999... → must land in bin 3 [0.3, 0.4)
            (0.6, 6),  # 0.6/0.1 == 5.9999... → must land in bin 6 [0.6, 0.7)
            (0.7, 7),  # 0.7/0.1 == 6.9999... → must land in bin 7 [0.7, 0.8)
            # p=1.0 must clamp to the last bin, not overflow.
            (1.0, 9),
            # Values strictly inside a bin are unaffected.
            (0.05, 0),
            (0.35, 3),
            (0.95, 9),
        ],
    )
    def test_exact_boundary_bin_placement(self, prob: float, expected_bin: int) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(prob, 1, question_id=f"q_{prob}"))
        curve = scorer.calibration_curve(n_bins=10)
        assert curve[expected_bin].count == 1, (
            f"p={prob} landed in bin {next(i for i, b in enumerate(curve) if b.count)}, "
            f"expected bin {expected_bin}"
        )
        assert sum(b.count for b in curve) == 1

    @pytest.mark.parametrize(
        "prob,expected_bin",
        [(i / 10, min(i, 9)) for i in range(11)],
    )
    def test_all_decimal_boundaries_are_low_inclusive(self, prob: float, expected_bin: int) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(prob, 1, question_id=f"boundary_{prob}"))
        curve = scorer.calibration_curve(n_bins=10)

        assert curve[expected_bin].count == 1, (
            f"p={prob} landed in bin {next(i for i, b in enumerate(curve) if b.count)}, "
            f"expected bin {expected_bin}"
        )
        assert sum(b.count for b in curve) == 1

    @pytest.mark.parametrize(
        "prob,expected_bin",
        [
            (0.2999999999, 2),
            (0.5999999999, 5),
            (0.6999999999, 6),
        ],
    )
    def test_values_below_boundary_remain_high_exclusive(
        self, prob: float, expected_bin: int
    ) -> None:
        scorer = ManifoldBrierScorer()
        scorer.add(_pred(prob, 1, question_id=f"below_boundary_{prob}"))
        curve = scorer.calibration_curve(n_bins=10)

        assert curve[expected_bin].count == 1, (
            f"p={prob} landed in bin {next(i for i, b in enumerate(curve) if b.count)}, "
            f"expected bin {expected_bin}"
        )
        assert sum(b.count for b in curve) == 1

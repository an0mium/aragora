"""Tests for aragora.metrics.manifold_brier — Manifold Brier scorer skeleton."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from aragora.metrics.manifold_brier import (
    BrierWindowSummary,
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

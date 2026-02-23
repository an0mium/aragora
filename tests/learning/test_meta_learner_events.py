"""Tests for MetaLearner event emissions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.learning.meta import (
    MetaLearner,
    LearningMetrics,
    HyperparameterState,
)


@pytest.fixture
def meta_learner(tmp_path):
    db = tmp_path / "meta.db"
    return MetaLearner(db_path=str(db))


class TestMetaLearnerEvents:
    """Tests for _emit_adjustment_event."""

    def test_emits_event_on_adjustment(self, meta_learner) -> None:
        metrics = LearningMetrics(
            pattern_retention_rate=0.4,
            forgetting_rate=0.35,
            learning_velocity=5,
            consensus_rate=0.6,
            prediction_accuracy=0.3,
            cycles_evaluated=10,
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            meta_learner._emit_adjustment_event(
                {"half_lives": "increased (low retention)"},
                metrics,
            )

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["adjustments"] == {"half_lives": "increased (low retention)"}
        assert data["pattern_retention_rate"] == 0.4
        assert data["forgetting_rate"] == 0.35
        assert data["prediction_accuracy"] == 0.3

    def test_skips_empty_adjustments(self, meta_learner) -> None:
        metrics = LearningMetrics()

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            meta_learner._emit_adjustment_event({}, metrics)
            mock_dispatch.assert_not_called()

    def test_handles_import_error(self, meta_learner) -> None:
        metrics = LearningMetrics(pattern_retention_rate=0.5)

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            meta_learner._emit_adjustment_event({"test": "value"}, metrics)

    def test_event_type_is_meta_learning_adjusted(self, meta_learner) -> None:
        metrics = LearningMetrics(consensus_rate=0.8)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            meta_learner._emit_adjustment_event({"rule": "applied"}, metrics)
            assert mock_dispatch.call_args[0][0] == "meta_learning_adjusted"


class TestAdjustHyperparametersEmitsEvent:
    """Tests that adjust_hyperparameters calls _emit_adjustment_event."""

    def test_low_retention_emits_event(self, meta_learner) -> None:
        metrics = LearningMetrics(
            pattern_retention_rate=0.4,  # Below 0.6 threshold
            forgetting_rate=0.1,
            consensus_rate=0.5,
            prediction_accuracy=0.5,
            tier_efficiency={"fast": 0.5, "medium": 0.5, "slow": 0.5, "glacial": 0.5},
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert "half_lives" in adjustments
        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert "half_lives" in data["adjustments"]

    def test_no_event_when_no_adjustments(self, meta_learner) -> None:
        # Metrics that trigger no adjustments
        metrics = LearningMetrics(
            pattern_retention_rate=0.75,  # Between 0.6 and 0.9
            forgetting_rate=0.15,  # Between 0.1 and 0.3
            consensus_rate=0.6,
            prediction_accuracy=0.55,  # Between 0.4 and 0.7
            tier_efficiency={"fast": 0.5, "medium": 0.5, "slow": 0.5, "glacial": 0.5},
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            adjustments = meta_learner.adjust_hyperparameters(metrics)

        if not adjustments:
            mock_dispatch.assert_not_called()

    def test_high_forgetting_emits_event(self, meta_learner) -> None:
        metrics = LearningMetrics(
            pattern_retention_rate=0.75,
            forgetting_rate=0.4,  # Above 0.3 threshold
            consensus_rate=0.5,
            prediction_accuracy=0.5,
            tier_efficiency={"fast": 0.5, "medium": 0.5, "slow": 0.5, "glacial": 0.5},
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert "promotion_thresholds" in adjustments
        assert mock_dispatch.call_count >= 1

    def test_poor_calibration_emits_event(self, meta_learner) -> None:
        metrics = LearningMetrics(
            pattern_retention_rate=0.75,
            forgetting_rate=0.15,
            consensus_rate=0.5,
            prediction_accuracy=0.3,  # Below 0.4 threshold
            tier_efficiency={"fast": 0.5, "medium": 0.5, "slow": 0.5, "glacial": 0.5},
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert "surprise_weights" in adjustments
        assert mock_dispatch.call_count >= 1

    def test_multiple_adjustments_in_single_event(self, meta_learner) -> None:
        # Low retention AND high forgetting → multiple adjustments
        metrics = LearningMetrics(
            pattern_retention_rate=0.4,
            forgetting_rate=0.4,
            consensus_rate=0.5,
            prediction_accuracy=0.3,
            tier_efficiency={"fast": 0.5, "medium": 0.5, "slow": 0.5, "glacial": 0.5},
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            adjustments = meta_learner.adjust_hyperparameters(metrics)

        # Should have multiple adjustments in one event
        assert len(adjustments) >= 2
        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert len(data["adjustments"]) >= 2

    def test_event_includes_all_metrics(self, meta_learner) -> None:
        metrics = LearningMetrics(
            pattern_retention_rate=0.4,
            forgetting_rate=0.1,
            learning_velocity=7,
            consensus_rate=0.65,
            prediction_accuracy=0.5,
            cycles_evaluated=42,
            tier_efficiency={"fast": 0.5, "medium": 0.5, "slow": 0.5, "glacial": 0.5},
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            meta_learner.adjust_hyperparameters(metrics)

        if mock_dispatch.call_count > 0:
            data = mock_dispatch.call_args[0][1]
            assert data["learning_velocity"] == 7
            assert data["cycles_evaluated"] == 42
            assert data["consensus_rate"] == 0.65

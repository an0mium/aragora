"""Tests for aragora.markets.scoring — Brier, payouts, calibration curve."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aragora.markets.scoring import (
    aggregate_brier,
    binary_outcome_value,
    brier_score,
    calibration_curve,
    evaluate_position_payout,
)
from aragora.markets.types import MarketPosition, ResolutionEvent


class TestBrierBasics:
    def test_perfect_prediction_scores_zero(self) -> None:
        assert brier_score(1.0, "yes") == 0.0
        assert brier_score(0.0, "no") == 0.0

    def test_maximally_wrong_prediction_scores_one(self) -> None:
        assert brier_score(0.0, "yes") == 1.0
        assert brier_score(1.0, "no") == 1.0

    def test_inconclusive_returns_none(self) -> None:
        assert brier_score(0.5, "inconclusive") is None
        assert binary_outcome_value("inconclusive") is None

    def test_calibrated_prior_scores_quarter(self) -> None:
        assert brier_score(0.5, "yes") == 0.25
        assert brier_score(0.5, "no") == 0.25


class TestAggregateBrier:
    def _make_position(
        self,
        *,
        agent: str,
        market: str,
        probability: float,
        stake: int = 10,
        submitted_at: datetime | None = None,
    ) -> MarketPosition:
        return MarketPosition.create(
            market_id=market,
            agent_id=agent,
            probability=probability,
            stake=stake,
            submitted_at=submitted_at or datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
        )

    def test_empty_input_returns_none_aggregates(self) -> None:
        report = aggregate_brier(
            agent_id="agent",
            positions=[],
            resolutions={},
        )
        assert report.scored_positions == 0
        assert report.mean_brier is None
        assert report.stake_weighted_brier is None
        assert report.decayed_brier is None

    def test_inconclusive_resolutions_do_not_contribute_to_score(self) -> None:
        positions = [
            self._make_position(agent="alice", market="mkt_1", probability=0.7),
            self._make_position(agent="alice", market="mkt_2", probability=0.4),
        ]
        resolutions = {
            "mkt_1": ResolutionEvent.yes(
                market_id="mkt_1",
                resolution_source="github_pr_state",
                resolved_at=datetime(2026, 4, 24, tzinfo=UTC),
            ),
            "mkt_2": ResolutionEvent.inconclusive(
                market_id="mkt_2",
                resolution_source="github_pr_state",
                resolved_at=datetime(2026, 4, 24, tzinfo=UTC),
            ),
        }
        report = aggregate_brier(
            agent_id="alice",
            positions=positions,
            resolutions=resolutions,
            now=datetime(2026, 4, 24, tzinfo=UTC),
            half_life_days=None,
        )
        assert report.scored_positions == 1
        assert report.inconclusive_positions == 1
        # Brier(0.7, yes) = 0.09
        assert report.mean_brier == pytest.approx(0.09)

    def test_stake_weighted_brier_emphasises_larger_stakes(self) -> None:
        positions = [
            self._make_position(agent="alice", market="mkt_1", probability=0.9, stake=80),
            self._make_position(
                agent="alice",
                market="mkt_2",
                probability=0.1,
                stake=20,
                submitted_at=datetime(2026, 4, 17, 12, 1, tzinfo=UTC),
            ),
        ]
        resolutions = {
            "mkt_1": ResolutionEvent.yes(
                market_id="mkt_1",
                resolution_source="x",
                resolved_at=datetime(2026, 4, 24, tzinfo=UTC),
            ),
            "mkt_2": ResolutionEvent.yes(
                market_id="mkt_2",
                resolution_source="x",
                resolved_at=datetime(2026, 4, 24, tzinfo=UTC),
            ),
        }
        report = aggregate_brier(
            agent_id="alice",
            positions=positions,
            resolutions=resolutions,
            now=datetime(2026, 4, 24, tzinfo=UTC),
            half_life_days=None,
        )
        # mean = (0.01 + 0.81) / 2 = 0.41
        # stake-weighted = (0.01*80 + 0.81*20) / 100 = (0.8 + 16.2) / 100 = 0.17
        assert report.mean_brier == pytest.approx(0.41)
        assert report.stake_weighted_brier == pytest.approx(0.17)

    def test_other_agents_positions_are_ignored(self) -> None:
        positions = [
            self._make_position(agent="alice", market="mkt_1", probability=0.9),
            self._make_position(agent="bob", market="mkt_1", probability=0.1),
        ]
        resolutions = {
            "mkt_1": ResolutionEvent.yes(market_id="mkt_1", resolution_source="x"),
        }
        report = aggregate_brier(agent_id="alice", positions=positions, resolutions=resolutions)
        assert report.scored_positions == 1

    def test_decay_weight_pulls_recent_above_old(self) -> None:
        old = self._make_position(
            agent="alice",
            market="mkt_1",
            probability=0.0,  # maximally wrong (Brier=1)
            stake=10,
            submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        recent = self._make_position(
            agent="alice",
            market="mkt_2",
            probability=1.0,  # perfectly right (Brier=0)
            stake=10,
            submitted_at=datetime(2026, 4, 16, tzinfo=UTC),
        )
        resolutions = {
            "mkt_1": ResolutionEvent.yes(
                market_id="mkt_1",
                resolution_source="x",
                resolved_at=datetime(2026, 1, 8, tzinfo=UTC),
            ),
            "mkt_2": ResolutionEvent.yes(
                market_id="mkt_2",
                resolution_source="x",
                resolved_at=datetime(2026, 4, 17, tzinfo=UTC),
            ),
        }
        report = aggregate_brier(
            agent_id="alice",
            positions=[old, recent],
            resolutions=resolutions,
            now=datetime(2026, 4, 17, tzinfo=UTC),
            half_life_days=30.0,
        )
        # Without decay: mean = 0.5, stake_weighted = 0.5
        # With 30d half-life: recent (Brier=0) weighs heavily; decayed should be < 0.25
        assert report.mean_brier == pytest.approx(0.5)
        assert report.decayed_brier is not None
        assert report.decayed_brier < 0.25


class TestPositionPayout:
    def test_perfect_prediction_returns_full_stake(self) -> None:
        position = MarketPosition.create(
            market_id="mkt_1", agent_id="alice", probability=1.0, stake=50
        )
        resolution = ResolutionEvent.yes(market_id="mkt_1", resolution_source="x")
        assert evaluate_position_payout(position=position, resolution=resolution) == 50

    def test_maximally_wrong_forfeits_full_stake(self) -> None:
        position = MarketPosition.create(
            market_id="mkt_1", agent_id="alice", probability=0.0, stake=50
        )
        resolution = ResolutionEvent.yes(market_id="mkt_1", resolution_source="x")
        assert evaluate_position_payout(position=position, resolution=resolution) == -50

    def test_calibrated_prior_breaks_even_to_zero(self) -> None:
        position = MarketPosition.create(
            market_id="mkt_1", agent_id="alice", probability=0.5, stake=50
        )
        resolution = ResolutionEvent.yes(market_id="mkt_1", resolution_source="x")
        # Brier=0.25 → payout fraction = 1 - 2*0.25 = 0.5 → 25 credits
        assert evaluate_position_payout(position=position, resolution=resolution) == 25

    def test_inconclusive_returns_zero(self) -> None:
        position = MarketPosition.create(
            market_id="mkt_1", agent_id="alice", probability=0.7, stake=50
        )
        resolution = ResolutionEvent.inconclusive(market_id="mkt_1", resolution_source="x")
        assert evaluate_position_payout(position=position, resolution=resolution) == 0


class TestCalibrationCurve:
    def test_buckets_aggregate_predicted_and_realized_means(self) -> None:
        positions = [
            MarketPosition.create(
                market_id=f"mkt_{i}",
                agent_id="alice",
                probability=p,
                stake=10,
                submitted_at=datetime(2026, 4, 17, 12, i, tzinfo=UTC),
            )
            for i, p in enumerate([0.05, 0.1, 0.7, 0.75, 0.95])
        ]
        resolutions = {
            "mkt_0": ResolutionEvent.no(market_id="mkt_0", resolution_source="x"),
            "mkt_1": ResolutionEvent.yes(market_id="mkt_1", resolution_source="x"),
            "mkt_2": ResolutionEvent.yes(market_id="mkt_2", resolution_source="x"),
            "mkt_3": ResolutionEvent.yes(market_id="mkt_3", resolution_source="x"),
            "mkt_4": ResolutionEvent.yes(market_id="mkt_4", resolution_source="x"),
        }
        curve = calibration_curve(positions=positions, resolutions=resolutions, bucket_count=10)
        # 0.05 -> bucket 0 (0.0-0.1), 0.1 -> bucket 1 (0.1-0.2),
        # 0.7 and 0.75 -> bucket 7 (0.7-0.8), 0.95 -> bucket 9 (0.9-1.0)
        buckets_present = [round(b["bucket_low"], 1) for b in curve]
        assert buckets_present == [0.0, 0.1, 0.7, 0.9]
        # Bucket 7 has 2 entries with realized mean 1.0 (both yes)
        seven_bucket = next(b for b in curve if round(b["bucket_low"], 1) == 0.7)
        assert seven_bucket["count"] == 2.0
        assert seven_bucket["realized_mean"] == 1.0

    def test_invalid_bucket_count_raises(self) -> None:
        with pytest.raises(ValueError):
            calibration_curve(positions=[], resolutions={}, bucket_count=1)

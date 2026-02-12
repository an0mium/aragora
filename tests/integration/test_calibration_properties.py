"""
ELO/calibration property tests.

Uses parametrize patterns (no Hypothesis dependency) to verify:
- ELO ratings are always positive
- Calibration scores are between 0-1
- Winning agents gain ELO, losing agents lose ELO
- Repeated wins increase confidence
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aragora.ranking.elo import AgentRating, EloSystem
from aragora.ranking.elo_core import calculate_new_elo, expected_score

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def elo_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def elo(elo_db):
    return EloSystem(elo_db)


# ---------------------------------------------------------------------------
# 1. ELO ratings stay positive
# ---------------------------------------------------------------------------


class TestEloPositivity:
    """ELO ratings must always be positive after any operation."""

    @pytest.mark.parametrize("initial_elo", [100, 500, 1000, 1500, 2000])
    def test_rating_after_loss_is_positive(self, elo, initial_elo):
        """Even after a loss, rating should remain positive."""
        elo.register_agent("winner")
        elo.register_agent("loser")

        # Set initial ratings manually
        rating = elo.get_rating("loser")
        rating.elo = initial_elo
        elo._save_rating(rating)

        elo.record_match(
            debate_id="test-loss",
            winner="winner",
            participants=["winner", "loser"],
            scores={"winner": 1.0, "loser": 0.0},
        )

        after = elo.get_rating("loser")
        assert after.elo > 0, f"ELO went non-positive: {after.elo}"

    @pytest.mark.parametrize("num_losses", [1, 5, 10, 20])
    def test_rating_positive_after_repeated_losses(self, elo, num_losses):
        """Rating stays positive even after many consecutive losses."""
        elo.register_agent("strong")
        elo.register_agent("weak")

        for i in range(num_losses):
            elo.record_match(
                debate_id=f"loss-{i}",
                winner="strong",
                participants=["strong", "weak"],
                scores={"strong": 1.0, "weak": 0.0},
            )

        rating = elo.get_rating("weak")
        assert rating.elo > 0

    def test_new_agent_starts_at_default_elo(self, elo):
        rating = elo.get_rating("new_agent")
        assert rating.elo == 1500  # DEFAULT_ELO (ARAGORA_ELO_INITIAL)
        assert rating.elo > 0


# ---------------------------------------------------------------------------
# 2. Calibration scores are bounded [0, 1]
# ---------------------------------------------------------------------------


class TestCalibrationBounds:
    """Calibration scores must be in [0, 1]."""

    @pytest.mark.parametrize(
        "correct,total,brier_sum",
        [
            (0, 10, 5.0),
            (5, 10, 2.5),
            (10, 10, 0.0),
            (3, 10, 7.0),
            (10, 10, 1.0),
        ],
    )
    def test_calibration_accuracy_bounded(self, correct, total, brier_sum):
        rating = AgentRating(
            agent_name="test",
            calibration_correct=correct,
            calibration_total=total,
            calibration_brier_sum=brier_sum,
        )

        assert 0.0 <= rating.calibration_accuracy <= 1.0

    @pytest.mark.parametrize(
        "correct,total,brier_sum",
        [
            (0, 10, 10.0),
            (5, 10, 5.0),
            (10, 10, 0.0),
            (8, 20, 4.0),
        ],
    )
    def test_calibration_brier_score_bounded(self, correct, total, brier_sum):
        rating = AgentRating(
            agent_name="test",
            calibration_correct=correct,
            calibration_total=total,
            calibration_brier_sum=brier_sum,
        )

        assert 0.0 <= rating.calibration_brier_score <= 1.0

    def test_calibration_score_zero_when_too_few_predictions(self):
        rating = AgentRating(
            agent_name="test",
            calibration_correct=3,
            calibration_total=4,
            calibration_brier_sum=0.5,
        )
        # calibration_score requires CALIBRATION_MIN_COUNT (default 5) predictions
        assert rating.calibration_score == 0.0

    @pytest.mark.parametrize("total", [10, 20, 50, 100])
    def test_calibration_score_bounded_with_enough_data(self, total):
        rating = AgentRating(
            agent_name="test",
            calibration_correct=total // 2,
            calibration_total=total,
            calibration_brier_sum=total * 0.25,
        )

        score = rating.calibration_score
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 3. Winners gain ELO, losers lose ELO
# ---------------------------------------------------------------------------


class TestEloDirectionality:
    """Winning should increase ELO, losing should decrease it."""

    def test_winner_gains_elo(self, elo):
        elo.register_agent("alice")
        elo.register_agent("bob")

        before = elo.get_rating("alice").elo

        elo.record_match(
            debate_id="directionality-test",
            winner="alice",
            participants=["alice", "bob"],
            scores={"alice": 1.0, "bob": 0.0},
        )

        after = elo.get_rating("alice").elo
        assert after > before, f"Winner ELO didn't increase: {before} -> {after}"

    def test_loser_loses_elo(self, elo):
        elo.register_agent("alice")
        elo.register_agent("bob")

        before = elo.get_rating("bob").elo

        elo.record_match(
            debate_id="directionality-test-2",
            winner="alice",
            participants=["alice", "bob"],
            scores={"alice": 1.0, "bob": 0.0},
        )

        after = elo.get_rating("bob").elo
        assert after < before, f"Loser ELO didn't decrease: {before} -> {after}"

    @pytest.mark.parametrize("num_wins", [1, 3, 5])
    def test_repeated_wins_increase_elo_monotonically(self, elo, num_wins):
        elo.register_agent("winner")
        elo.register_agent("loser")

        ratings = [elo.get_rating("winner").elo]

        for i in range(num_wins):
            elo.record_match(
                debate_id=f"win-{i}",
                winner="winner",
                participants=["winner", "loser"],
                scores={"winner": 1.0, "loser": 0.0},
            )
            ratings.append(elo.get_rating("winner").elo)

        # Each successive rating should be higher
        for j in range(1, len(ratings)):
            assert ratings[j] > ratings[j - 1], (
                f"ELO not monotonically increasing at step {j}: {ratings}"
            )

    def test_draw_keeps_elos_close(self, elo):
        elo.register_agent("alice")
        elo.register_agent("bob")

        before_a = elo.get_rating("alice").elo
        before_b = elo.get_rating("bob").elo

        elo.record_match(
            debate_id="draw-test",
            participants=["alice", "bob"],
            scores={"alice": 0.5, "bob": 0.5},
        )

        after_a = elo.get_rating("alice").elo
        after_b = elo.get_rating("bob").elo

        # In a draw between equal-rated agents, changes should be minimal
        assert abs(after_a - before_a) < 5
        assert abs(after_b - before_b) < 5


# ---------------------------------------------------------------------------
# 4. Win/loss record tracking
# ---------------------------------------------------------------------------


class TestWinLossTracking:
    """Verify win/loss/draw counters are accurate."""

    def test_win_count_increments(self, elo):
        elo.register_agent("alice")
        elo.register_agent("bob")

        for i in range(3):
            elo.record_match(
                debate_id=f"track-{i}",
                winner="alice",
                participants=["alice", "bob"],
                scores={"alice": 1.0, "bob": 0.0},
            )

        rating = elo.get_rating("alice")
        assert rating.wins >= 3

    def test_loss_count_increments(self, elo):
        elo.register_agent("alice")
        elo.register_agent("bob")

        for i in range(3):
            elo.record_match(
                debate_id=f"loss-track-{i}",
                winner="alice",
                participants=["alice", "bob"],
                scores={"alice": 1.0, "bob": 0.0},
            )

        rating = elo.get_rating("bob")
        assert rating.losses >= 3

    def test_win_rate_bounded(self, elo):
        elo.register_agent("alice")
        elo.register_agent("bob")

        elo.record_match(
            debate_id="wr-1",
            winner="alice",
            participants=["alice", "bob"],
            scores={"alice": 1.0, "bob": 0.0},
        )
        elo.record_match(
            debate_id="wr-2",
            winner="bob",
            participants=["alice", "bob"],
            scores={"bob": 1.0, "alice": 0.0},
        )

        alice = elo.get_rating("alice")
        bob = elo.get_rating("bob")
        assert 0.0 <= alice.win_rate <= 1.0
        assert 0.0 <= bob.win_rate <= 1.0


# ---------------------------------------------------------------------------
# 5. ELO calculation purity
# ---------------------------------------------------------------------------


class TestEloCalculation:
    """Test the pure ELO math functions directly."""

    @pytest.mark.parametrize(
        "rating_a,rating_b",
        [
            (1000, 1000),
            (1200, 800),
            (800, 1200),
            (500, 1500),
            (2000, 1000),
        ],
    )
    def test_expected_scores_sum_to_one(self, rating_a, rating_b):
        ea = expected_score(rating_a, rating_b)
        eb = expected_score(rating_b, rating_a)
        assert abs(ea + eb - 1.0) < 1e-6

    @pytest.mark.parametrize("k", [16, 24, 32, 48])
    def test_new_elo_is_deterministic(self, k):
        # Verify the ELO math is deterministic and returns a float
        new = calculate_new_elo(
            current_elo=100,
            expected=0.99,
            actual=0.0,
            k=k,
        )
        assert isinstance(new, float)
        # Same inputs should give same output
        new2 = calculate_new_elo(current_elo=100, expected=0.99, actual=0.0, k=k)
        assert new == new2

    def test_higher_rated_player_has_higher_expected_score(self):
        e_strong = expected_score(1500, 1000)
        e_weak = expected_score(1000, 1500)

        assert e_strong > 0.5
        assert e_weak < 0.5
        assert e_strong > e_weak

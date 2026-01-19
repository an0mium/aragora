"""Tests for match recording engine."""

import pytest
import json
from unittest.mock import MagicMock, patch

from aragora.ranking.match_recorder import (
    build_match_scores,
    generate_match_id,
    normalize_match_params,
    compute_calibration_k_multipliers,
    save_match,
    check_duplicate_match,
    determine_winner,
)


class TestBuildMatchScores:
    """Test build_match_scores function."""

    def test_win_scores(self):
        """Test scores for a win."""
        scores = build_match_scores("claude", "gpt", is_draw=False)

        assert scores["claude"] == 1.0
        assert scores["gpt"] == 0.0

    def test_draw_scores(self):
        """Test scores for a draw."""
        scores = build_match_scores("claude", "gpt", is_draw=True)

        assert scores["claude"] == 0.5
        assert scores["gpt"] == 0.5

    def test_win_scores_order_preserved(self):
        """Test winner is first in scores regardless of name."""
        scores = build_match_scores("zeta", "alpha", is_draw=False)

        assert scores["zeta"] == 1.0
        assert scores["alpha"] == 0.0


class TestGenerateMatchId:
    """Test generate_match_id function."""

    def test_basic_id(self):
        """Test basic match ID generation."""
        match_id = generate_match_id(["claude", "gpt"])

        assert "claude-vs-gpt" in match_id
        assert "debate" in match_id
        # UUID suffix
        assert len(match_id.split("-")[-1]) == 8

    def test_id_with_task(self):
        """Test match ID with task label."""
        match_id = generate_match_id(["claude", "gpt"], task="rate-limiting")

        assert "rate-limiting" in match_id
        assert "claude-vs-gpt" in match_id

    def test_id_with_domain(self):
        """Test match ID with domain label."""
        match_id = generate_match_id(["claude", "gpt"], domain="security")

        assert "security" in match_id
        assert "claude-vs-gpt" in match_id

    def test_id_task_takes_precedence(self):
        """Test task takes precedence over domain."""
        match_id = generate_match_id(
            ["claude", "gpt"], task="rate-limiting", domain="security"
        )

        assert "rate-limiting" in match_id
        # Domain should not appear when task is provided

    def test_id_empty_participants(self):
        """Test match ID with empty participants."""
        match_id = generate_match_id([])

        assert "match" in match_id
        assert "debate" in match_id

    def test_id_multiple_participants(self):
        """Test match ID with multiple participants."""
        match_id = generate_match_id(["claude", "gpt", "gemini"])

        assert "claude-vs-gpt-vs-gemini" in match_id

    def test_id_is_unique(self):
        """Test that generated IDs are unique."""
        id1 = generate_match_id(["claude", "gpt"])
        id2 = generate_match_id(["claude", "gpt"])

        assert id1 != id2  # Different UUID suffixes


class TestNormalizeMatchParams:
    """Test normalize_match_params function."""

    def test_modern_signature_with_scores(self):
        """Test modern signature with explicit scores."""
        debate_id, participants, scores = normalize_match_params(
            debate_id="d1",
            participants=["claude", "gpt"],
            scores={"claude": 1.0, "gpt": 0.0},
            winner=None,
            loser=None,
            draw=None,
            task=None,
            domain=None,
        )

        assert debate_id == "d1"
        assert participants == ["claude", "gpt"]
        assert scores == {"claude": 1.0, "gpt": 0.0}

    def test_modern_signature_with_winner_loser(self):
        """Test modern signature with winner/loser."""
        debate_id, participants, scores = normalize_match_params(
            debate_id="d1",
            participants=None,
            scores=None,
            winner="claude",
            loser="gpt",
            draw=False,
            task=None,
            domain=None,
        )

        assert debate_id == "d1"
        assert participants == ["claude", "gpt"]
        assert scores == {"claude": 1.0, "gpt": 0.0}

    def test_modern_signature_draw_with_participants(self):
        """Test modern signature for draw."""
        debate_id, participants, scores = normalize_match_params(
            debate_id="d1",
            participants=["claude", "gpt"],
            scores=None,
            winner=None,
            loser=None,
            draw=True,
            task=None,
            domain=None,
        )

        assert scores == {"claude": 0.5, "gpt": 0.5}

    def test_modern_signature_infers_participants_from_scores(self):
        """Test participants inferred from scores."""
        debate_id, participants, scores = normalize_match_params(
            debate_id="d1",
            participants=None,
            scores={"claude": 1.0, "gpt": 0.0},
            winner=None,
            loser=None,
            draw=None,
            task=None,
            domain=None,
        )

        assert set(participants) == {"claude", "gpt"}

    def test_modern_signature_generates_id(self):
        """Test debate_id is generated when not provided."""
        debate_id, participants, scores = normalize_match_params(
            debate_id=None,
            participants=["claude", "gpt"],
            scores={"claude": 1.0, "gpt": 0.0},
            winner=None,
            loser=None,
            draw=None,
            task=None,
            domain=None,
        )

        assert debate_id != ""
        assert "claude-vs-gpt" in debate_id

    def test_legacy_signature_string_participants(self):
        """Test legacy signature where participants is loser name."""
        debate_id, participants, scores = normalize_match_params(
            debate_id="claude",  # Legacy: winner name
            participants="gpt",  # Legacy: loser name (string)
            scores=None,
            winner=None,
            loser=None,
            draw=None,
            task=None,
            domain=None,
        )

        assert participants == ["claude", "gpt"]
        assert scores == {"claude": 1.0, "gpt": 0.0}
        # New ID generated
        assert "claude-vs-gpt" in debate_id

    def test_legacy_signature_draw(self):
        """Test legacy signature with draw."""
        debate_id, participants, scores = normalize_match_params(
            debate_id="claude",
            participants="gpt",
            scores=True,  # Legacy: draw flag as bool
            winner=None,
            loser=None,
            draw=None,
            task=None,
            domain=None,
        )

        assert scores == {"claude": 0.5, "gpt": 0.5}

    def test_legacy_signature_explicit_draw(self):
        """Test legacy signature with explicit draw flag."""
        debate_id, participants, scores = normalize_match_params(
            debate_id="claude",
            participants="gpt",
            scores=None,
            winner=None,
            loser=None,
            draw=True,
            task=None,
            domain=None,
        )

        assert scores == {"claude": 0.5, "gpt": 0.5}

    def test_legacy_signature_with_explicit_winner_loser(self):
        """Test legacy signature with explicit winner/loser."""
        debate_id, participants, scores = normalize_match_params(
            debate_id=None,
            participants="ignored",
            scores=None,
            winner="claude",
            loser="gpt",
            draw=False,
            task=None,
            domain=None,
        )

        assert "claude" in participants
        assert "gpt" in participants

    def test_legacy_signature_missing_names_raises(self):
        """Test legacy signature with missing names raises."""
        with pytest.raises(ValueError) as exc:
            normalize_match_params(
                debate_id=None,
                participants="loser_only",  # String = legacy
                scores=None,
                winner=None,  # No winner
                loser=None,
                draw=None,
                task=None,
                domain=None,
            )
        assert "winner and loser must be provided" in str(exc.value)


class TestComputeCalibrationKMultipliers:
    """Test compute_calibration_k_multipliers function."""

    def test_no_tracker_returns_empty(self):
        """Test returns empty dict without tracker."""
        result = compute_calibration_k_multipliers(["claude", "gpt"], None)
        assert result == {}

    def test_well_calibrated(self):
        """Test multiplier for well-calibrated agent (ECE < 0.1)."""
        mock_tracker = MagicMock()
        mock_tracker.get_expected_calibration_error.return_value = 0.05

        result = compute_calibration_k_multipliers(["claude"], mock_tracker)

        assert result["claude"] == 1.0

    def test_slightly_miscalibrated(self):
        """Test multiplier for slightly miscalibrated agent (ECE 0.1-0.2)."""
        mock_tracker = MagicMock()
        mock_tracker.get_expected_calibration_error.return_value = 0.15

        result = compute_calibration_k_multipliers(["claude"], mock_tracker)

        assert result["claude"] == 1.1

    def test_moderately_miscalibrated(self):
        """Test multiplier for moderately miscalibrated agent (ECE 0.2-0.3)."""
        mock_tracker = MagicMock()
        mock_tracker.get_expected_calibration_error.return_value = 0.25

        result = compute_calibration_k_multipliers(["claude"], mock_tracker)

        assert result["claude"] == 1.25

    def test_poorly_calibrated(self):
        """Test multiplier for poorly calibrated agent (ECE > 0.3)."""
        mock_tracker = MagicMock()
        mock_tracker.get_expected_calibration_error.return_value = 0.5

        result = compute_calibration_k_multipliers(["claude"], mock_tracker)

        assert result["claude"] == 1.4

    def test_boundary_values(self):
        """Test boundary ECE values."""
        mock_tracker = MagicMock()

        # At 0.1 boundary - should be 1.1 (>= 0.1 is slightly miscalibrated)
        mock_tracker.get_expected_calibration_error.return_value = 0.1
        result = compute_calibration_k_multipliers(["agent"], mock_tracker)
        assert result["agent"] == 1.1

        # At 0.2 boundary - should be 1.25 (>= 0.2 is moderately miscalibrated)
        mock_tracker.get_expected_calibration_error.return_value = 0.2
        result = compute_calibration_k_multipliers(["agent"], mock_tracker)
        assert result["agent"] == 1.25

        # At 0.3 boundary - should be 1.4 (>= 0.3 is poorly calibrated)
        mock_tracker.get_expected_calibration_error.return_value = 0.3
        result = compute_calibration_k_multipliers(["agent"], mock_tracker)
        assert result["agent"] == 1.4

    def test_multiple_agents(self):
        """Test multipliers for multiple agents."""
        mock_tracker = MagicMock()

        def get_ece(agent):
            return {"claude": 0.05, "gpt": 0.25, "gemini": 0.5}[agent]

        mock_tracker.get_expected_calibration_error.side_effect = get_ece

        result = compute_calibration_k_multipliers(["claude", "gpt", "gemini"], mock_tracker)

        assert result["claude"] == 1.0  # Well calibrated
        assert result["gpt"] == 1.25  # Moderately miscalibrated
        assert result["gemini"] == 1.4  # Poorly calibrated

    def test_error_handling(self):
        """Test error handling for failed lookups."""
        mock_tracker = MagicMock()
        mock_tracker.get_expected_calibration_error.side_effect = KeyError("Unknown agent")

        result = compute_calibration_k_multipliers(["unknown"], mock_tracker)

        # Should default to 1.0 on error
        assert result["unknown"] == 1.0

    def test_attribute_error_handling(self):
        """Test handling of AttributeError."""
        mock_tracker = MagicMock()
        mock_tracker.get_expected_calibration_error.side_effect = AttributeError()

        result = compute_calibration_k_multipliers(["agent"], mock_tracker)

        assert result["agent"] == 1.0


class TestSaveMatch:
    """Test save_match function."""

    def test_save_match(self):
        """Test saving a match."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        save_match(
            db=mock_db,
            debate_id="d1",
            winner="claude",
            participants=["claude", "gpt"],
            domain="security",
            scores={"claude": 1.0, "gpt": 0.0},
            elo_changes={"claude": 15.0, "gpt": -15.0},
        )

        mock_cursor.execute.assert_called_once()
        args = mock_cursor.execute.call_args[0]

        assert "INSERT OR REPLACE INTO matches" in args[0]
        values = args[1]
        assert values[0] == "d1"
        assert values[1] == "claude"
        assert json.loads(values[2]) == ["claude", "gpt"]
        assert values[3] == "security"
        assert json.loads(values[4]) == {"claude": 1.0, "gpt": 0.0}
        assert json.loads(values[5]) == {"claude": 15.0, "gpt": -15.0}
        mock_conn.commit.assert_called_once()

    def test_save_match_draw(self):
        """Test saving a draw."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        save_match(
            db=mock_db,
            debate_id="d1",
            winner=None,  # Draw
            participants=["claude", "gpt"],
            domain=None,
            scores={"claude": 0.5, "gpt": 0.5},
            elo_changes={"claude": 0.0, "gpt": 0.0},
        )

        args = mock_cursor.execute.call_args[0]
        values = args[1]
        assert values[1] is None  # No winner


class TestCheckDuplicateMatch:
    """Test check_duplicate_match function."""

    def test_no_duplicate(self):
        """Test when match doesn't exist."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchone.return_value = None

        result = check_duplicate_match(mock_db, "d1")

        assert result is None

    def test_duplicate_found(self):
        """Test when match exists."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchone.return_value = ('{"claude": 15, "gpt": -15}',)

        result = check_duplicate_match(mock_db, "d1")

        assert result == {"claude": 15, "gpt": -15}

    def test_duplicate_invalid_json(self):
        """Test handling of invalid JSON in stored data."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchone.return_value = ("invalid json",)

        result = check_duplicate_match(mock_db, "d1")

        # Should return default empty dict
        assert result == {}


class TestDetermineWinner:
    """Test determine_winner function."""

    def test_clear_winner(self):
        """Test determining clear winner."""
        winner = determine_winner({"claude": 1.0, "gpt": 0.0})
        assert winner == "claude"

    def test_draw(self):
        """Test determining draw (tied scores)."""
        winner = determine_winner({"claude": 0.5, "gpt": 0.5})
        assert winner is None

    def test_multiple_participants_winner(self):
        """Test winner from multiple participants."""
        winner = determine_winner({
            "claude": 0.8,
            "gpt": 0.6,
            "gemini": 0.4,
        })
        assert winner == "claude"

    def test_multiple_participants_tie_for_first(self):
        """Test tie for first place."""
        winner = determine_winner({
            "claude": 0.8,
            "gpt": 0.8,
            "gemini": 0.4,
        })
        # Tie between top two = no winner
        assert winner is None

    def test_single_participant(self):
        """Test single participant is winner."""
        winner = determine_winner({"claude": 1.0})
        assert winner == "claude"

    def test_empty_scores(self):
        """Test empty scores."""
        winner = determine_winner({})
        assert winner is None

    def test_close_scores(self):
        """Test very close but different scores."""
        winner = determine_winner({"claude": 0.51, "gpt": 0.49})
        assert winner == "claude"


class TestMatchRecorderIntegration:
    """Integration tests for match recording workflow."""

    def test_full_match_workflow(self):
        """Test complete match recording workflow."""
        # 1. Generate match ID
        match_id = generate_match_id(["claude", "gpt"], task="rate-limiting")
        assert "rate-limiting" in match_id

        # 2. Build scores
        scores = build_match_scores("claude", "gpt", is_draw=False)
        assert scores["claude"] == 1.0

        # 3. Determine winner
        winner = determine_winner(scores)
        assert winner == "claude"

        # 4. Normalize params (modern style)
        debate_id, participants, final_scores = normalize_match_params(
            debate_id=match_id,
            participants=["claude", "gpt"],
            scores=scores,
            winner=None,
            loser=None,
            draw=None,
            task=None,
            domain=None,
        )
        assert debate_id == match_id
        assert final_scores == scores

    def test_legacy_to_modern_conversion(self):
        """Test converting legacy API calls to modern format."""
        # Legacy style call
        debate_id, participants, scores = normalize_match_params(
            debate_id="winner_name",
            participants="loser_name",  # String = legacy
            scores=None,
            winner=None,
            loser=None,
            draw=False,
            task="test",
            domain=None,
        )

        # Should be converted to modern format
        assert participants == ["winner_name", "loser_name"]
        assert scores == {"winner_name": 1.0, "loser_name": 0.0}
        assert "test" in debate_id

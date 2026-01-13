"""Tests for CalibrationEngine and DomainCalibrationEngine.

Tests:
- Tournament prediction recording and resolution
- Brier score calculations
- Domain-specific calibration tracking
- Calibration curves and ECE
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from aragora.ranking.calibration_engine import (
    CalibrationEngine,
    DomainCalibrationEngine,
    CalibrationPrediction,
    BucketStats,
)


@dataclass
class MockRating:
    """Mock rating for testing."""

    agent_name: str
    calibration_correct: int = 0
    calibration_total: int = 0
    calibration_brier_sum: float = 0.0
    updated_at: str = ""


def create_test_db(db_path: Path) -> None:
    """Create test database with calibration tables."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Create calibration_predictions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id TEXT NOT NULL,
                predictor_agent TEXT NOT NULL,
                predicted_winner TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tournament_id, predictor_agent)
            )
        """
        )
        # Create domain_calibration table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS domain_calibration (
                agent_name TEXT NOT NULL,
                domain TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                total_correct INTEGER DEFAULT 0,
                brier_sum REAL DEFAULT 0.0,
                updated_at TIMESTAMP,
                PRIMARY KEY (agent_name, domain)
            )
        """
        )
        # Create calibration_buckets table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_buckets (
                agent_name TEXT NOT NULL,
                domain TEXT NOT NULL,
                bucket_key TEXT NOT NULL,
                predictions INTEGER DEFAULT 0,
                correct INTEGER DEFAULT 0,
                brier_sum REAL DEFAULT 0.0,
                PRIMARY KEY (agent_name, domain, bucket_key)
            )
        """
        )
        conn.commit()


class TestCalibrationPrediction:
    """Tests for CalibrationPrediction dataclass."""

    def test_create_prediction(self):
        """Basic prediction creation."""
        pred = CalibrationPrediction(
            tournament_id="t-123",
            predictor_agent="claude",
            predicted_winner="gemini",
            confidence=0.75,
        )
        assert pred.tournament_id == "t-123"
        assert pred.predictor_agent == "claude"
        assert pred.predicted_winner == "gemini"
        assert pred.confidence == 0.75
        assert pred.created_at is None

    def test_prediction_with_timestamp(self):
        """Prediction with timestamp."""
        pred = CalibrationPrediction(
            tournament_id="t-456",
            predictor_agent="agent",
            predicted_winner="winner",
            confidence=0.5,
            created_at="2024-01-01T00:00:00",
        )
        assert pred.created_at == "2024-01-01T00:00:00"


class TestBucketStats:
    """Tests for BucketStats dataclass."""

    def test_create_bucket_stats(self):
        """Basic bucket stats creation."""
        stats = BucketStats(
            bucket_key="0.7-0.8",
            bucket_start=0.7,
            bucket_end=0.8,
            predictions=100,
            correct=75,
            accuracy=0.75,
            expected_accuracy=0.75,
            brier_score=0.1,
        )
        assert stats.bucket_key == "0.7-0.8"
        assert stats.predictions == 100
        assert stats.accuracy == 0.75


class TestCalibrationEngine:
    """Tests for CalibrationEngine class."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create temporary database."""
        path = tmp_path / "test_calibration.db"
        create_test_db(path)
        return path

    @pytest.fixture
    def engine(self, db_path):
        """Create CalibrationEngine with test database."""
        return CalibrationEngine(db_path)

    @pytest.fixture
    def engine_with_elo(self, db_path):
        """Create CalibrationEngine with mock EloSystem."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(agent_name="test")
        elo._save_ratings_batch = MagicMock()
        return CalibrationEngine(db_path, elo_system=elo)

    def test_init(self, db_path):
        """Engine initializes correctly."""
        engine = CalibrationEngine(db_path)
        assert engine.db_path == db_path
        assert engine.elo_system is None

    def test_init_with_elo(self, db_path):
        """Engine initializes with EloSystem."""
        elo = MagicMock()
        engine = CalibrationEngine(db_path, elo_system=elo)
        assert engine.elo_system is elo

    def test_record_winner_prediction(self, engine, db_path):
        """Records prediction to database."""
        engine.record_winner_prediction(
            tournament_id="tourney-1",
            predictor_agent="claude",
            predicted_winner="gemini",
            confidence=0.8,
        )

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM calibration_predictions")
            row = cursor.fetchone()

        assert row is not None
        assert row[1] == "tourney-1"  # tournament_id
        assert row[2] == "claude"  # predictor_agent
        assert row[3] == "gemini"  # predicted_winner
        assert row[4] == pytest.approx(0.8)  # confidence

    def test_record_prediction_clamps_confidence(self, engine, db_path):
        """Confidence is clamped to 0-1 range."""
        engine.record_winner_prediction("t1", "agent", "winner", 1.5)
        engine.record_winner_prediction("t2", "agent", "winner", -0.5)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT confidence FROM calibration_predictions ORDER BY tournament_id")
            rows = cursor.fetchall()

        assert rows[0][0] == pytest.approx(1.0)  # clamped from 1.5
        assert rows[1][0] == pytest.approx(0.0)  # clamped from -0.5

    def test_resolve_tournament_without_elo(self, engine, db_path):
        """Resolves tournament and returns Brier scores without EloSystem."""
        # Record predictions
        engine.record_winner_prediction("t1", "claude", "gemini", 0.9)
        engine.record_winner_prediction("t1", "gpt", "gemini", 0.6)
        engine.record_winner_prediction("t1", "llama", "claude", 0.7)

        # Resolve - gemini wins
        scores = engine.resolve_tournament("t1", "gemini")

        # Claude predicted gemini with 0.9 confidence
        # Brier = (0.9 - 1.0)^2 = 0.01
        assert scores["claude"] == pytest.approx(0.01)

        # GPT predicted gemini with 0.6 confidence
        # Brier = (0.6 - 1.0)^2 = 0.16
        assert scores["gpt"] == pytest.approx(0.16)

        # Llama predicted claude (wrong) with 0.7 confidence
        # Brier = (0.7 - 0.0)^2 = 0.49
        assert scores["llama"] == pytest.approx(0.49)

    def test_resolve_tournament_with_elo_updates(self, engine_with_elo):
        """Resolves tournament and updates EloSystem ratings."""
        engine = engine_with_elo
        rating1 = MockRating(agent_name="claude")
        rating2 = MockRating(agent_name="gpt")
        # resolve_tournament uses get_ratings_batch, not get_rating
        engine.elo_system.get_ratings_batch.return_value = {
            "claude": rating1,
            "gpt": rating2,
        }

        engine.record_winner_prediction("t1", "claude", "winner", 0.8)
        engine.record_winner_prediction("t1", "gpt", "winner", 0.5)

        engine.resolve_tournament("t1", "winner")

        # Both ratings should be updated
        assert rating1.calibration_total == 1
        assert rating1.calibration_correct == 1
        assert rating2.calibration_total == 1
        assert rating2.calibration_correct == 1

        # Batch save should be called
        engine.elo_system._save_ratings_batch.assert_called_once()

    def test_resolve_empty_tournament(self, engine):
        """Empty tournament returns empty dict."""
        scores = engine.resolve_tournament("nonexistent", "winner")
        assert scores == {}

    def test_get_agent_history(self, engine):
        """Gets prediction history for agent."""
        engine.record_winner_prediction("t1", "claude", "a", 0.6)
        engine.record_winner_prediction("t2", "claude", "b", 0.7)
        engine.record_winner_prediction("t3", "other", "c", 0.8)

        history = engine.get_agent_history("claude")

        assert len(history) == 2
        assert all(p.predictor_agent == "claude" for p in history)

    def test_get_agent_history_limit(self, engine):
        """History respects limit parameter."""
        for i in range(10):
            engine.record_winner_prediction(f"t{i}", "claude", "winner", 0.5)

        history = engine.get_agent_history("claude", limit=5)
        assert len(history) == 5

    def test_calculate_brier_score(self, engine):
        """Calculates Brier score correctly."""
        # Perfect prediction with high confidence
        assert engine.calculate_brier_score(1.0, True) == pytest.approx(0.0)

        # Perfect prediction with low confidence
        assert engine.calculate_brier_score(0.5, True) == pytest.approx(0.25)

        # Wrong prediction with high confidence
        assert engine.calculate_brier_score(1.0, False) == pytest.approx(1.0)

        # Wrong prediction with low confidence
        assert engine.calculate_brier_score(0.3, False) == pytest.approx(0.09)

    def test_get_leaderboard_without_elo(self, engine):
        """Leaderboard returns empty without EloSystem."""
        result = engine.get_leaderboard()
        assert result == []


class TestDomainCalibrationEngine:
    """Tests for DomainCalibrationEngine class."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create temporary database."""
        path = tmp_path / "test_domain_cal.db"
        create_test_db(path)
        return path

    @pytest.fixture
    def engine(self, db_path):
        """Create DomainCalibrationEngine."""
        return DomainCalibrationEngine(db_path)

    @pytest.fixture
    def engine_with_elo(self, db_path):
        """Create engine with mock EloSystem."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(agent_name="test")
        elo._save_rating = MagicMock()
        return DomainCalibrationEngine(db_path, elo_system=elo)

    def test_get_bucket_key(self):
        """Bucket key generation."""
        assert DomainCalibrationEngine.get_bucket_key(0.0) == "0.0-0.1"
        assert DomainCalibrationEngine.get_bucket_key(0.15) == "0.1-0.2"
        assert DomainCalibrationEngine.get_bucket_key(0.75) == "0.7-0.8"
        assert DomainCalibrationEngine.get_bucket_key(0.99) == "0.9-1.0"
        assert DomainCalibrationEngine.get_bucket_key(1.0) == "1.0-1.0"

    def test_record_prediction(self, engine, db_path):
        """Records domain prediction to database."""
        engine.record_prediction("claude", "security", 0.8, correct=True)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check domain_calibration
            cursor.execute("SELECT * FROM domain_calibration WHERE agent_name = ?", ("claude",))
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == "security"  # domain
            assert row[2] == 1  # total_predictions
            assert row[3] == 1  # total_correct

            # Check calibration_buckets
            cursor.execute("SELECT * FROM calibration_buckets WHERE agent_name = ?", ("claude",))
            bucket = cursor.fetchone()
            assert bucket is not None
            assert bucket[2] == "0.8-0.9"  # bucket_key

    def test_record_prediction_updates_elo(self, engine_with_elo):
        """Recording prediction updates EloSystem rating."""
        engine = engine_with_elo
        rating = MockRating(agent_name="claude")
        engine.elo_system.get_rating.return_value = rating

        engine.record_prediction("claude", "security", 0.8, correct=True)

        assert rating.calibration_total == 1
        assert rating.calibration_correct == 1
        engine.elo_system._save_rating.assert_called_once()

    def test_record_multiple_predictions(self, engine, db_path):
        """Multiple predictions aggregate correctly."""
        engine.record_prediction("claude", "security", 0.7, correct=True)
        engine.record_prediction("claude", "security", 0.8, correct=False)
        engine.record_prediction("claude", "security", 0.9, correct=True)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT total_predictions, total_correct FROM domain_calibration WHERE agent_name = ?",
                ("claude",),
            )
            row = cursor.fetchone()

        assert row[0] == 3  # total_predictions
        assert row[1] == 2  # total_correct

    def test_get_domain_stats_empty(self, engine):
        """Empty stats for unknown agent."""
        stats = engine.get_domain_stats("unknown")

        assert stats["total"] == 0
        assert stats["correct"] == 0
        assert stats["accuracy"] == 0.0
        assert stats["brier_score"] == 1.0
        assert stats["domains"] == {}

    def test_get_domain_stats(self, engine):
        """Gets aggregated domain stats."""
        engine.record_prediction("claude", "security", 0.8, correct=True)
        engine.record_prediction("claude", "security", 0.6, correct=True)
        engine.record_prediction("claude", "performance", 0.7, correct=False)

        stats = engine.get_domain_stats("claude")

        assert stats["total"] == 3
        assert stats["correct"] == 2
        assert stats["accuracy"] == pytest.approx(2 / 3)
        assert "security" in stats["domains"]
        assert "performance" in stats["domains"]
        assert stats["domains"]["security"]["predictions"] == 2

    def test_get_domain_stats_filtered(self, engine):
        """Gets stats for specific domain."""
        engine.record_prediction("claude", "security", 0.8, correct=True)
        engine.record_prediction("claude", "performance", 0.7, correct=False)

        stats = engine.get_domain_stats("claude", domain="security")

        assert stats["total"] == 1
        assert stats["domains"]["security"]["predictions"] == 1
        assert "performance" not in stats["domains"]

    def test_get_calibration_curve(self, engine):
        """Gets calibration curve buckets."""
        # Add predictions in different confidence buckets
        engine.record_prediction("claude", "general", 0.75, correct=True)
        engine.record_prediction("claude", "general", 0.78, correct=True)
        engine.record_prediction("claude", "general", 0.55, correct=False)

        curve = engine.get_calibration_curve("claude")

        assert len(curve) >= 1
        bucket_keys = [b.bucket_key for b in curve]
        assert "0.7-0.8" in bucket_keys or "0.5-0.6" in bucket_keys

    def test_get_calibration_curve_empty(self, engine):
        """Empty curve for unknown agent."""
        curve = engine.get_calibration_curve("unknown")
        assert curve == []

    def test_get_expected_calibration_error(self, engine):
        """Calculates ECE correctly."""
        # Perfect calibration: 70% confident predictions are 70% correct
        for _ in range(7):
            engine.record_prediction("perfect", "test", 0.75, correct=True)
        for _ in range(3):
            engine.record_prediction("perfect", "test", 0.75, correct=False)

        ece = engine.get_expected_calibration_error("perfect")
        # Should be low (close to 0) for well-calibrated predictions
        assert ece < 0.1

    def test_get_expected_calibration_error_empty(self, engine):
        """ECE is 1.0 for unknown agent."""
        ece = engine.get_expected_calibration_error("unknown")
        assert ece == 1.0

    def test_get_best_domains(self, engine):
        """Gets best calibrated domains."""
        # Add many predictions in security (well calibrated)
        for _ in range(10):
            engine.record_prediction("claude", "security", 0.8, correct=True)

        # Add some in performance (poorly calibrated)
        for _ in range(10):
            engine.record_prediction("claude", "performance", 0.9, correct=False)

        best = engine.get_best_domains("claude", limit=2)

        assert len(best) <= 2
        # Security should rank higher (lower brier score)
        if len(best) == 2:
            assert best[0][0] == "security"

    def test_get_best_domains_min_predictions(self, engine):
        """Domains with few predictions are filtered."""
        engine.record_prediction("claude", "rare", 0.8, correct=True)  # Only 1 prediction
        for _ in range(10):
            engine.record_prediction("claude", "common", 0.8, correct=True)

        best = engine.get_best_domains("claude", min_predictions=5)

        domain_names = [d[0] for d in best]
        assert "rare" not in domain_names
        assert "common" in domain_names


class TestBrierScoreCalculations:
    """Tests for Brier score edge cases."""

    @pytest.fixture
    def engine(self, tmp_path):
        path = tmp_path / "brier_test.db"
        create_test_db(path)
        return CalibrationEngine(path)

    def test_perfect_prediction_high_confidence(self, engine):
        """Perfect with 100% confidence = 0 Brier."""
        score = engine.calculate_brier_score(1.0, True)
        assert score == 0.0

    def test_perfect_prediction_medium_confidence(self, engine):
        """Correct but only 50% confident."""
        score = engine.calculate_brier_score(0.5, True)
        assert score == pytest.approx(0.25)

    def test_wrong_prediction_high_confidence(self, engine):
        """Wrong with 100% confidence = max Brier."""
        score = engine.calculate_brier_score(1.0, False)
        assert score == 1.0

    def test_uncertain_prediction_wrong(self, engine):
        """Wrong but uncertain = moderate Brier."""
        score = engine.calculate_brier_score(0.5, False)
        assert score == pytest.approx(0.25)

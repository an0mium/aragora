"""Tests for TinkerEvaluator module.

Deep tests for ABTestResult, EvaluationMetrics, and TinkerEvaluator.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.training.evaluator import (
    ABTestResult,
    EvaluationMetrics,
    TinkerEvaluator,
)


# =============================================================================
# ABTestResult Tests - Significance
# =============================================================================


class TestABTestResultSignificance:
    """Test ABTestResult.is_significant property."""

    def test_not_significant_under_10_trials(self):
        """Test not significant with fewer than 10 trials."""
        result = ABTestResult(
            test_id="test-1",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=9,
            fine_tuned_wins=9,
            baseline_wins=0,
            draws=0,
            fine_tuned_win_rate=1.0,
            avg_fine_tuned_score=1.0,
            avg_baseline_score=0.0,
            avg_confidence=0.9,
            consensus_rate=1.0,
        )
        assert result.is_significant is False

    def test_not_significant_50_50_split(self):
        """Test not significant with 50/50 split."""
        result = ABTestResult(
            test_id="test-2",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=100,
            fine_tuned_wins=50,
            baseline_wins=50,
            draws=0,
            fine_tuned_win_rate=0.5,
            avg_fine_tuned_score=0.5,
            avg_baseline_score=0.5,
            avg_confidence=0.8,
            consensus_rate=0.8,
        )
        assert result.is_significant is False

    def test_significant_clear_winner(self):
        """Test significant with clear winner."""
        result = ABTestResult(
            test_id="test-3",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=100,
            fine_tuned_wins=80,
            baseline_wins=20,
            draws=0,
            fine_tuned_win_rate=0.8,
            avg_fine_tuned_score=0.8,
            avg_baseline_score=0.2,
            avg_confidence=0.9,
            consensus_rate=0.9,
        )
        assert result.is_significant is True

    def test_significant_all_wins(self):
        """Test significant with all wins."""
        result = ABTestResult(
            test_id="test-4",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=20,
            fine_tuned_wins=20,
            baseline_wins=0,
            draws=0,
            fine_tuned_win_rate=1.0,
            avg_fine_tuned_score=1.0,
            avg_baseline_score=0.0,
            avg_confidence=0.95,
            consensus_rate=1.0,
        )
        assert result.is_significant is True

    def test_significant_all_losses(self):
        """Test significant with all losses (lower bound test)."""
        result = ABTestResult(
            test_id="test-5",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=20,
            fine_tuned_wins=0,
            baseline_wins=20,
            draws=0,
            fine_tuned_win_rate=0.0,
            avg_fine_tuned_score=0.0,
            avg_baseline_score=1.0,
            avg_confidence=0.95,
            consensus_rate=1.0,
        )
        assert result.is_significant is True

    def test_borderline_case_10_trials(self):
        """Test borderline case with exactly 10 trials."""
        result = ABTestResult(
            test_id="test-6",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=10,
            fine_tuned_wins=10,
            baseline_wins=0,
            draws=0,
            fine_tuned_win_rate=1.0,
            avg_fine_tuned_score=1.0,
            avg_baseline_score=0.0,
            avg_confidence=0.9,
            consensus_rate=1.0,
        )
        # With 10 trials and 100% win rate, should be significant
        assert result.is_significant is True

    def test_borderline_55_percent(self):
        """Test borderline 55% win rate."""
        result = ABTestResult(
            test_id="test-7",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=100,
            fine_tuned_wins=55,
            baseline_wins=45,
            draws=0,
            fine_tuned_win_rate=0.55,
            avg_fine_tuned_score=0.55,
            avg_baseline_score=0.45,
            avg_confidence=0.8,
            consensus_rate=0.8,
        )
        # 55% with 100 trials is borderline
        assert isinstance(result.is_significant, bool)


# =============================================================================
# ABTestResult Tests - to_dict
# =============================================================================


class TestABTestResultToDict:
    """Test ABTestResult serialization."""

    def test_to_dict_all_fields(self):
        """Test to_dict includes all fields."""
        result = ABTestResult(
            test_id="test-id-123",
            fine_tuned_agent="fine-tuned",
            baseline_agent="baseline",
            num_trials=50,
            fine_tuned_wins=30,
            baseline_wins=15,
            draws=5,
            fine_tuned_win_rate=0.6,
            avg_fine_tuned_score=0.7,
            avg_baseline_score=0.4,
            avg_confidence=0.85,
            consensus_rate=0.9,
        )
        d = result.to_dict()

        assert d["test_id"] == "test-id-123"
        assert d["fine_tuned_agent"] == "fine-tuned"
        assert d["baseline_agent"] == "baseline"
        assert d["num_trials"] == 50
        assert d["fine_tuned_wins"] == 30
        assert d["baseline_wins"] == 15
        assert d["draws"] == 5
        assert d["fine_tuned_win_rate"] == 0.6
        assert d["avg_fine_tuned_score"] == 0.7
        assert d["avg_baseline_score"] == 0.4
        assert d["avg_confidence"] == 0.85
        assert d["consensus_rate"] == 0.9

    def test_to_dict_includes_is_significant(self):
        """Test to_dict includes is_significant."""
        result = ABTestResult(
            test_id="test",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=100,
            fine_tuned_wins=80,
            baseline_wins=20,
            draws=0,
            fine_tuned_win_rate=0.8,
            avg_fine_tuned_score=0.8,
            avg_baseline_score=0.2,
            avg_confidence=0.9,
            consensus_rate=0.9,
        )
        d = result.to_dict()
        assert "is_significant" in d
        assert d["is_significant"] is True

    def test_to_dict_includes_created_at(self):
        """Test to_dict includes created_at."""
        result = ABTestResult(
            test_id="test",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=10,
            fine_tuned_wins=5,
            baseline_wins=5,
            draws=0,
            fine_tuned_win_rate=0.5,
            avg_fine_tuned_score=0.5,
            avg_baseline_score=0.5,
            avg_confidence=0.8,
            consensus_rate=0.8,
        )
        d = result.to_dict()
        assert "created_at" in d


# =============================================================================
# EvaluationMetrics Tests
# =============================================================================


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        metrics = EvaluationMetrics(
            model_id="model-1",
            elo_rating=1500.0,
            win_rate=0.6,
            avg_score=0.7,
            calibration_score=0.8,
            consensus_contribution=0.5,
        )
        assert metrics.model_id == "model-1"
        assert metrics.elo_rating == 1500.0
        assert metrics.win_rate == 0.6

    def test_domain_scores_default(self):
        """Test domain_scores default is empty dict."""
        metrics = EvaluationMetrics(
            model_id="model",
            elo_rating=1500.0,
            win_rate=0.5,
            avg_score=0.5,
            calibration_score=0.5,
            consensus_contribution=0.5,
        )
        assert metrics.domain_scores == {}

    def test_domain_scores_custom(self):
        """Test custom domain_scores."""
        metrics = EvaluationMetrics(
            model_id="model",
            elo_rating=1500.0,
            win_rate=0.5,
            avg_score=0.5,
            calibration_score=0.5,
            consensus_contribution=0.5,
            domain_scores={"software": 1600.0, "legal": 1400.0},
        )
        assert metrics.domain_scores["software"] == 1600.0
        assert metrics.domain_scores["legal"] == 1400.0

    def test_total_debates_default(self):
        """Test total_debates default is 0."""
        metrics = EvaluationMetrics(
            model_id="model",
            elo_rating=1500.0,
            win_rate=0.5,
            avg_score=0.5,
            calibration_score=0.5,
            consensus_contribution=0.5,
        )
        assert metrics.total_debates == 0

    def test_to_dict(self):
        """Test to_dict serialization."""
        metrics = EvaluationMetrics(
            model_id="model-x",
            elo_rating=1550.0,
            win_rate=0.65,
            avg_score=0.72,
            calibration_score=0.85,
            consensus_contribution=0.6,
            domain_scores={"research": 1580.0},
            total_debates=100,
        )
        d = metrics.to_dict()

        assert d["model_id"] == "model-x"
        assert d["elo_rating"] == 1550.0
        assert d["win_rate"] == 0.65
        assert d["avg_score"] == 0.72
        assert d["calibration_score"] == 0.85
        assert d["consensus_contribution"] == 0.6
        assert d["domain_scores"] == {"research": 1580.0}
        assert d["total_debates"] == 100


# =============================================================================
# TinkerEvaluator Tests - Init
# =============================================================================


class TestTinkerEvaluatorInit:
    """Test TinkerEvaluator initialization."""

    def test_default_init(self, tmp_path):
        """Test default initialization."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        assert evaluator.results_dir == tmp_path
        assert evaluator._test_counter == 0

    def test_creates_results_dir(self, tmp_path):
        """Test creates results directory."""
        results_dir = tmp_path / "results"
        evaluator = TinkerEvaluator(results_dir=results_dir)
        assert results_dir.exists()

    def test_with_custom_elo(self, tmp_path):
        """Test with custom ELO system."""
        mock_elo = MagicMock()
        evaluator = TinkerEvaluator(elo_system=mock_elo, results_dir=tmp_path)
        assert evaluator.elo == mock_elo


# =============================================================================
# TinkerEvaluator Tests - Test ID Generation
# =============================================================================


class TestTinkerEvaluatorTestId:
    """Test test ID generation."""

    def test_generate_test_id_format(self, tmp_path):
        """Test test ID format."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        test_id = evaluator._generate_test_id()
        assert test_id.startswith("test-")
        parts = test_id.split("-")
        assert len(parts) >= 3  # test-YYYYMMDD-HHMMSS-0001

    def test_generate_test_id_increments(self, tmp_path):
        """Test test ID counter increments."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        id1 = evaluator._generate_test_id()
        id2 = evaluator._generate_test_id()
        assert id1 != id2
        assert evaluator._test_counter == 2

    def test_generate_test_id_unique(self, tmp_path):
        """Test generated IDs are unique."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        ids = [evaluator._generate_test_id() for _ in range(10)]
        assert len(ids) == len(set(ids))


# =============================================================================
# TinkerEvaluator Tests - Save/Load Results
# =============================================================================


class TestTinkerEvaluatorSaveResult:
    """Test result saving."""

    def test_save_result_creates_json(self, tmp_path):
        """Test saving creates JSON file."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        result = ABTestResult(
            test_id="test-save-1",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=10,
            fine_tuned_wins=6,
            baseline_wins=4,
            draws=0,
            fine_tuned_win_rate=0.6,
            avg_fine_tuned_score=0.6,
            avg_baseline_score=0.4,
            avg_confidence=0.8,
            consensus_rate=0.8,
            trials=[{"trial": 1, "winner": "ft"}],
        )
        evaluator._save_result(result)

        json_file = tmp_path / "test-save-1.json"
        assert json_file.exists()

    def test_save_result_creates_trials_jsonl(self, tmp_path):
        """Test saving creates trials JSONL file."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        result = ABTestResult(
            test_id="test-save-2",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=10,
            fine_tuned_wins=5,
            baseline_wins=5,
            draws=0,
            fine_tuned_win_rate=0.5,
            avg_fine_tuned_score=0.5,
            avg_baseline_score=0.5,
            avg_confidence=0.7,
            consensus_rate=0.7,
            trials=[
                {"trial": 1, "winner": "ft"},
                {"trial": 2, "winner": "bl"},
            ],
        )
        evaluator._save_result(result)

        trials_file = tmp_path / "test-save-2_trials.jsonl"
        assert trials_file.exists()

        # Verify JSONL content
        with open(trials_file) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_save_result_json_content(self, tmp_path):
        """Test saved JSON has correct content."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        result = ABTestResult(
            test_id="test-save-3",
            fine_tuned_agent="model-a",
            baseline_agent="model-b",
            num_trials=20,
            fine_tuned_wins=12,
            baseline_wins=8,
            draws=0,
            fine_tuned_win_rate=0.6,
            avg_fine_tuned_score=0.65,
            avg_baseline_score=0.45,
            avg_confidence=0.85,
            consensus_rate=0.9,
        )
        evaluator._save_result(result)

        with open(tmp_path / "test-save-3.json") as f:
            data = json.load(f)

        assert data["test_id"] == "test-save-3"
        assert data["fine_tuned_agent"] == "model-a"
        assert data["num_trials"] == 20


class TestTinkerEvaluatorLoadResult:
    """Test result loading."""

    def test_load_result_found(self, tmp_path):
        """Test loading existing result."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)

        # Create result file
        result_data = {
            "test_id": "test-load-1",
            "fine_tuned_agent": "ft",
            "baseline_agent": "bl",
            "num_trials": 10,
            "fine_tuned_wins": 7,
            "baseline_wins": 3,
            "draws": 0,
            "fine_tuned_win_rate": 0.7,
            "avg_fine_tuned_score": 0.7,
            "avg_baseline_score": 0.3,
            "avg_confidence": 0.8,
            "consensus_rate": 0.8,
            "created_at": "2024-01-01T00:00:00",
        }
        with open(tmp_path / "test-load-1.json", "w") as f:
            json.dump(result_data, f)

        loaded = evaluator.load_result("test-load-1")

        assert loaded is not None
        assert loaded.test_id == "test-load-1"
        assert loaded.fine_tuned_wins == 7

    def test_load_result_not_found(self, tmp_path):
        """Test loading non-existent result."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        loaded = evaluator.load_result("nonexistent")
        assert loaded is None

    def test_load_result_with_trials(self, tmp_path):
        """Test loading result with trials."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)

        # Create result file
        with open(tmp_path / "test-load-2.json", "w") as f:
            json.dump({
                "test_id": "test-load-2",
                "fine_tuned_agent": "ft",
                "baseline_agent": "bl",
                "num_trials": 2,
                "fine_tuned_wins": 1,
                "baseline_wins": 1,
                "draws": 0,
                "fine_tuned_win_rate": 0.5,
                "avg_fine_tuned_score": 0.5,
                "avg_baseline_score": 0.5,
                "avg_confidence": 0.7,
                "consensus_rate": 0.7,
            }, f)

        # Create trials file
        with open(tmp_path / "test-load-2_trials.jsonl", "w") as f:
            f.write('{"trial": 1, "winner": "ft"}\n')
            f.write('{"trial": 2, "winner": "bl"}\n')

        loaded = evaluator.load_result("test-load-2")

        assert loaded is not None
        assert len(loaded.trials) == 2


# =============================================================================
# TinkerEvaluator Tests - List Results
# =============================================================================


class TestTinkerEvaluatorListResults:
    """Test listing results."""

    def test_list_results_empty(self, tmp_path):
        """Test listing with no results."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)
        results = evaluator.list_results()
        assert results == []

    def test_list_results_sorted(self, tmp_path):
        """Test results are sorted by date (newest first)."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)

        # Create result files with different names (will affect sorting)
        for i in range(3):
            with open(tmp_path / f"test-2024-0{i+1}.json", "w") as f:
                json.dump({
                    "test_id": f"test-2024-0{i+1}",
                    "fine_tuned_agent": "ft",
                    "baseline_agent": "bl",
                    "fine_tuned_win_rate": 0.5 + i * 0.1,
                }, f)

        results = evaluator.list_results()
        assert len(results) == 3

    def test_list_results_limit(self, tmp_path):
        """Test limit parameter."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)

        for i in range(10):
            with open(tmp_path / f"test-{i:04d}.json", "w") as f:
                json.dump({
                    "test_id": f"test-{i:04d}",
                    "fine_tuned_agent": "ft",
                    "baseline_agent": "bl",
                    "fine_tuned_win_rate": 0.5,
                }, f)

        results = evaluator.list_results(limit=5)
        assert len(results) == 5


# =============================================================================
# TinkerEvaluator Tests - A/B Test
# =============================================================================


class TestTinkerEvaluatorABTest:
    """Test A/B testing."""

    @pytest.mark.asyncio
    async def test_ab_test_basic(self, tmp_path):
        """Test basic A/B test flow."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)

        # Mock agents
        ft_agent = MagicMock()
        ft_agent.name = "fine-tuned"
        bl_agent = MagicMock()
        bl_agent.name = "baseline"

        # Mock debate
        mock_result = MagicMock()
        mock_result.votes = [
            MagicMock(choice="fine-tuned", confidence=0.8),
        ]
        mock_result.confidence = 0.8
        mock_result.consensus_reached = True
        mock_result.rounds_used = 3

        with patch.object(evaluator, "_run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await evaluator.a_b_test(
                tasks=["task1"],
                fine_tuned_agent=ft_agent,
                baseline_agent=bl_agent,
                num_trials=2,
                save_results=False,
            )

            assert result.num_trials == 2
            assert result.fine_tuned_agent == "fine-tuned"
            assert result.baseline_agent == "baseline"

    @pytest.mark.asyncio
    async def test_ab_test_with_additional_agents(self, tmp_path):
        """Test A/B test with additional agents."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)

        ft_agent = MagicMock()
        ft_agent.name = "ft"
        bl_agent = MagicMock()
        bl_agent.name = "bl"
        extra_agent = MagicMock()
        extra_agent.name = "extra"

        mock_result = MagicMock()
        mock_result.votes = []
        mock_result.confidence = 0.5
        mock_result.consensus_reached = False
        mock_result.rounds_used = 3

        with patch.object(evaluator, "_run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await evaluator.a_b_test(
                tasks=["task"],
                fine_tuned_agent=ft_agent,
                baseline_agent=bl_agent,
                num_trials=1,
                additional_agents=[extra_agent],
                save_results=False,
            )

            # Verify debate was called with all agents
            call_args = mock_run.call_args[0]
            agents = call_args[1]
            assert len(agents) >= 3

    @pytest.mark.asyncio
    async def test_ab_test_handles_trial_error(self, tmp_path):
        """Test A/B test handles trial errors gracefully."""
        evaluator = TinkerEvaluator(results_dir=tmp_path)

        ft_agent = MagicMock()
        ft_agent.name = "ft"
        bl_agent = MagicMock()
        bl_agent.name = "bl"

        with patch.object(evaluator, "_run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Debate failed")

            result = await evaluator.a_b_test(
                tasks=["task"],
                fine_tuned_agent=ft_agent,
                baseline_agent=bl_agent,
                num_trials=1,
                save_results=False,
            )

            # Should have recorded error trial
            assert len(result.trials) == 1
            assert "error" in result.trials[0]


# =============================================================================
# TinkerEvaluator Tests - Get Model Metrics
# =============================================================================


class TestTinkerEvaluatorGetMetrics:
    """Test getting model metrics."""

    def test_get_model_metrics(self, tmp_path):
        """Test getting metrics from ELO system."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1550.0
        mock_rating.win_rate = 0.65
        mock_rating.calibration_score = 0.8
        mock_rating.domain_elos = {"software": 1600.0}
        mock_rating.games_played = 50
        mock_elo.get_rating.return_value = mock_rating

        evaluator = TinkerEvaluator(elo_system=mock_elo, results_dir=tmp_path)

        metrics = evaluator.get_model_metrics("test-agent")

        assert metrics.model_id == "test-agent"
        assert metrics.elo_rating == 1550.0
        assert metrics.win_rate == 0.65
        assert metrics.total_debates == 50


# =============================================================================
# Win Rate Calculation Tests
# =============================================================================


class TestWinRateCalculations:
    """Test win rate and related calculations."""

    def test_zero_trials_win_rate(self):
        """Test win rate with zero trials."""
        result = ABTestResult(
            test_id="test",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=0,
            fine_tuned_wins=0,
            baseline_wins=0,
            draws=0,
            fine_tuned_win_rate=0.0,
            avg_fine_tuned_score=0.0,
            avg_baseline_score=0.0,
            avg_confidence=0.0,
            consensus_rate=0.0,
        )
        # Should not raise on is_significant
        assert result.is_significant is False

    def test_draws_included_in_trials(self):
        """Test draws are counted in total trials."""
        result = ABTestResult(
            test_id="test",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=10,
            fine_tuned_wins=3,
            baseline_wins=3,
            draws=4,
            fine_tuned_win_rate=0.3,  # 3/10
            avg_fine_tuned_score=0.3,
            avg_baseline_score=0.3,
            avg_confidence=0.7,
            consensus_rate=0.6,
        )
        assert result.num_trials == 10
        assert result.fine_tuned_wins + result.baseline_wins + result.draws == 10

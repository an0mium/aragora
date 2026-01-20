"""Tests for training module.

Tests the training system including:
- TrainingScheduler: job scheduling, state management
- TrainingJob: job lifecycle, serialization
- TinkerEvaluator: A/B testing, metrics
- ABTestResult: statistical significance
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.training.training_scheduler import (
    JobStatus,
    JobType,
    SchedulerConfig,
    TrainingJob,
    TrainingScheduler,
)
from aragora.training.evaluator import (
    ABTestResult,
    EvaluationMetrics,
    TinkerEvaluator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def scheduler_config(temp_dir):
    """Create a scheduler config with temp directories."""
    return SchedulerConfig(
        data_dir=temp_dir / "data",
        checkpoint_dir=temp_dir / "checkpoints",
        max_concurrent_jobs=2,
        export_limit=100,
    )


@pytest.fixture
def scheduler(scheduler_config):
    """Create a training scheduler."""
    return TrainingScheduler(config=scheduler_config)


@pytest.fixture
def mock_tinker_client():
    """Create a mock Tinker client."""
    client = AsyncMock()

    # Mock training result
    from aragora.training.tinker_client import TrainingResult, TrainingState

    result = MagicMock(spec=TrainingResult)
    result.job_id = "tinker-job-123"
    result.model_id = "model-123"
    result.state = TrainingState.COMPLETED
    result.error_message = None

    client.train_sft.return_value = result
    client.train_dpo.return_value = result
    client.close = AsyncMock()

    return client


@pytest.fixture
def evaluator(temp_dir):
    """Create a Tinker evaluator."""
    return TinkerEvaluator(results_dir=temp_dir / "results")


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    from aragora.core import Agent

    agent = MagicMock(spec=Agent)
    agent.name = "test-agent"
    return agent


@pytest.fixture
def mock_debate_result():
    """Create a mock debate result."""
    from aragora.core import DebateResult, Vote

    vote1 = MagicMock(spec=Vote)
    vote1.choice = "fine-tuned"
    vote1.confidence = 0.8

    vote2 = MagicMock(spec=Vote)
    vote2.choice = "baseline"
    vote2.confidence = 0.6

    result = MagicMock(spec=DebateResult)
    result.consensus_reached = True
    result.confidence = 0.75
    result.votes = [vote1, vote2]
    result.rounds_used = 3

    return result


# =============================================================================
# JobType and JobStatus Tests
# =============================================================================


class TestJobEnums:
    """Test job type and status enums."""

    def test_job_type_values(self):
        """Test JobType enum values."""
        assert JobType.SFT.value == "sft"
        assert JobType.DPO.value == "dpo"
        assert JobType.GAUNTLET.value == "gauntlet"
        assert JobType.COMBINED.value == "combined"

    def test_job_status_values(self):
        """Test JobStatus enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PREPARING.value == "preparing"
        assert JobStatus.SUBMITTED.value == "submitted"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"


# =============================================================================
# TrainingJob Tests
# =============================================================================


class TestTrainingJob:
    """Test TrainingJob dataclass."""

    def test_create_job(self):
        """Test creating a training job."""
        job = TrainingJob(
            job_id="job-001",
            job_type=JobType.SFT,
            model="llama-3.3-70b",
        )

        assert job.job_id == "job-001"
        assert job.job_type == JobType.SFT
        assert job.model == "llama-3.3-70b"
        assert job.status == JobStatus.PENDING
        assert job.created_at is not None

    def test_job_with_config(self):
        """Test job with custom config."""
        job = TrainingJob(
            job_id="job-002",
            job_type=JobType.DPO,
            model="model",
            config={
                "adapter_name": "my-adapter",
                "beta": 0.15,
            },
        )

        assert job.config["adapter_name"] == "my-adapter"
        assert job.config["beta"] == 0.15

    def test_job_to_dict(self):
        """Test job serialization."""
        job = TrainingJob(
            job_id="job-003",
            job_type=JobType.COMBINED,
            model="model",
            status=JobStatus.COMPLETED,
            tinker_job_id="tinker-123",
            model_id="model-456",
        )

        data = job.to_dict()

        assert data["job_id"] == "job-003"
        assert data["job_type"] == "combined"
        assert data["status"] == "completed"
        assert data["tinker_job_id"] == "tinker-123"
        assert data["model_id"] == "model-456"

    def test_job_timestamps(self):
        """Test job timestamps."""
        job = TrainingJob(
            job_id="job-004",
            job_type=JobType.SFT,
            model="model",
        )

        # Should have created_at
        assert job.created_at is not None
        datetime.fromisoformat(job.created_at)

        # Should not have started/completed
        assert job.started_at is None
        assert job.completed_at is None


# =============================================================================
# SchedulerConfig Tests
# =============================================================================


class TestSchedulerConfig:
    """Test SchedulerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SchedulerConfig()

        assert config.max_concurrent_jobs == 1
        assert config.sft_min_confidence == 0.7
        assert config.dpo_min_elo_difference == 50.0
        assert config.export_limit == 1000
        assert config.replay_data_ratio == 0.2

    def test_custom_config(self):
        """Test custom configuration."""
        config = SchedulerConfig(
            max_concurrent_jobs=4,
            export_limit=500,
            sft_min_confidence=0.8,
        )

        assert config.max_concurrent_jobs == 4
        assert config.export_limit == 500
        assert config.sft_min_confidence == 0.8


# =============================================================================
# TrainingScheduler Tests - Job Creation
# =============================================================================


class TestTrainingSchedulerJobCreation:
    """Test job creation in TrainingScheduler."""

    def test_generate_job_id(self, scheduler):
        """Test job ID generation."""
        id1 = scheduler._generate_job_id()
        id2 = scheduler._generate_job_id()

        assert id1.startswith("job-")
        assert id2.startswith("job-")
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_schedule_sft(self, scheduler, mock_tinker_client):
        """Test scheduling SFT job."""
        scheduler._client = mock_tinker_client

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(
                model="llama-3.3-70b",
                adapter_name="test-adapter",
            )

            assert job is not None
            assert job.job_type == JobType.SFT
            assert job.config["adapter_name"] == "test-adapter"
            assert job.job_id in scheduler._jobs

    @pytest.mark.asyncio
    async def test_schedule_dpo(self, scheduler, mock_tinker_client):
        """Test scheduling DPO job."""
        scheduler._client = mock_tinker_client

        with patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            job = await scheduler.schedule_dpo(
                model="llama-3.3-70b",
                beta=0.15,
            )

            assert job is not None
            assert job.job_type == JobType.DPO
            assert job.config["beta"] == 0.15

    @pytest.mark.asyncio
    async def test_schedule_combined(self, scheduler, mock_tinker_client):
        """Test scheduling combined job."""
        scheduler._client = mock_tinker_client

        with patch.object(scheduler, "_run_combined_job", new_callable=AsyncMock):
            job = await scheduler.schedule_combined(
                model="llama-3.3-70b",
                adapter_name="combined-adapter",
            )

            assert job is not None
            assert job.job_type == JobType.COMBINED


# =============================================================================
# TrainingScheduler Tests - Job Management
# =============================================================================


class TestTrainingSchedulerJobManagement:
    """Test job management in TrainingScheduler."""

    def test_get_job(self, scheduler):
        """Test getting job by ID."""
        job = TrainingJob(
            job_id="test-job",
            job_type=JobType.SFT,
            model="model",
        )
        scheduler._jobs["test-job"] = job

        retrieved = scheduler.get_job("test-job")
        assert retrieved is job

        assert scheduler.get_job("nonexistent") is None

    def test_list_jobs(self, scheduler):
        """Test listing jobs."""
        for i in range(5):
            job = TrainingJob(
                job_id=f"job-{i}",
                job_type=JobType.SFT,
                model="model",
                status=JobStatus.COMPLETED if i % 2 == 0 else JobStatus.FAILED,
            )
            scheduler._jobs[f"job-{i}"] = job

        # All jobs
        all_jobs = scheduler.list_jobs()
        assert len(all_jobs) == 5

        # Filter by status
        completed = scheduler.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed) == 3

        # Limit
        limited = scheduler.list_jobs(limit=2)
        assert len(limited) == 2

    def test_list_jobs_by_type(self, scheduler):
        """Test listing jobs by type."""
        scheduler._jobs["sft-job"] = TrainingJob(job_id="sft-job", job_type=JobType.SFT, model="m")
        scheduler._jobs["dpo-job"] = TrainingJob(job_id="dpo-job", job_type=JobType.DPO, model="m")

        sft_jobs = scheduler.list_jobs(job_type=JobType.SFT)
        assert len(sft_jobs) == 1
        assert sft_jobs[0].job_id == "sft-job"

    def test_cancel_job_pending(self, scheduler):
        """Test cancelling pending job."""
        job = TrainingJob(
            job_id="cancel-me",
            job_type=JobType.SFT,
            model="model",
            status=JobStatus.PENDING,
        )
        scheduler._jobs["cancel-me"] = job

        result = scheduler.cancel_job("cancel-me")

        assert result is True
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None

    def test_cancel_job_running(self, scheduler):
        """Test cannot cancel running job."""
        job = TrainingJob(
            job_id="running",
            job_type=JobType.SFT,
            model="model",
            status=JobStatus.RUNNING,
        )
        scheduler._jobs["running"] = job

        result = scheduler.cancel_job("running")

        assert result is False
        assert job.status == JobStatus.RUNNING

    def test_cancel_nonexistent_job(self, scheduler):
        """Test cancelling nonexistent job."""
        assert scheduler.cancel_job("nonexistent") is False


# =============================================================================
# TrainingScheduler Tests - State Persistence
# =============================================================================


class TestTrainingSchedulerPersistence:
    """Test scheduler state persistence."""

    def test_save_state(self, scheduler, temp_dir):
        """Test saving scheduler state."""
        job = TrainingJob(
            job_id="save-test",
            job_type=JobType.SFT,
            model="model",
            status=JobStatus.COMPLETED,
        )
        scheduler._jobs["save-test"] = job
        scheduler._job_counter = 5

        state_path = temp_dir / "state.json"
        scheduler.save_state(state_path)

        assert state_path.exists()

        with open(state_path) as f:
            data = json.load(f)

        assert data["job_counter"] == 5
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_id"] == "save-test"

    def test_load_state(self, scheduler, temp_dir):
        """Test loading scheduler state."""
        state = {
            "job_counter": 10,
            "jobs": [
                {
                    "job_id": "loaded-job",
                    "job_type": "sft",
                    "model": "model",
                    "status": "completed",
                    "config": {"key": "value"},
                },
            ],
            "saved_at": datetime.now().isoformat(),
        }

        state_path = temp_dir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f)

        scheduler.load_state(state_path)

        assert scheduler._job_counter == 10
        assert "loaded-job" in scheduler._jobs
        assert scheduler._jobs["loaded-job"].job_type == JobType.SFT

    def test_load_state_missing_file(self, scheduler, temp_dir):
        """Test loading from missing file does nothing."""
        scheduler._job_counter = 0
        scheduler.load_state(temp_dir / "nonexistent.json")
        assert scheduler._job_counter == 0


# =============================================================================
# ABTestResult Tests
# =============================================================================


class TestABTestResult:
    """Test ABTestResult dataclass."""

    def test_create_result(self):
        """Test creating AB test result."""
        result = ABTestResult(
            test_id="test-001",
            fine_tuned_agent="ft-agent",
            baseline_agent="bl-agent",
            num_trials=20,
            fine_tuned_wins=12,
            baseline_wins=6,
            draws=2,
            fine_tuned_win_rate=0.6,
            avg_fine_tuned_score=0.75,
            avg_baseline_score=0.55,
            avg_confidence=0.8,
            consensus_rate=0.9,
        )

        assert result.test_id == "test-001"
        assert result.fine_tuned_wins == 12
        assert result.fine_tuned_win_rate == 0.6

    def test_is_significant_true(self):
        """Test significance detection for clear win."""
        result = ABTestResult(
            test_id="sig",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=100,
            fine_tuned_wins=80,
            baseline_wins=15,
            draws=5,
            fine_tuned_win_rate=0.8,
            avg_fine_tuned_score=0.8,
            avg_baseline_score=0.4,
            avg_confidence=0.85,
            consensus_rate=0.9,
        )

        assert result.is_significant is True

    def test_is_significant_false_small_sample(self):
        """Test significance with small sample size."""
        result = ABTestResult(
            test_id="small",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=5,
            fine_tuned_wins=3,
            baseline_wins=2,
            draws=0,
            fine_tuned_win_rate=0.6,
            avg_fine_tuned_score=0.6,
            avg_baseline_score=0.5,
            avg_confidence=0.7,
            consensus_rate=0.8,
        )

        assert result.is_significant is False

    def test_is_significant_false_close_result(self):
        """Test significance with close results."""
        result = ABTestResult(
            test_id="close",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=50,
            fine_tuned_wins=26,
            baseline_wins=24,
            draws=0,
            fine_tuned_win_rate=0.52,
            avg_fine_tuned_score=0.52,
            avg_baseline_score=0.48,
            avg_confidence=0.75,
            consensus_rate=0.85,
        )

        assert result.is_significant is False

    def test_to_dict(self):
        """Test ABTestResult serialization."""
        result = ABTestResult(
            test_id="test-serialize",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=10,
            fine_tuned_wins=7,
            baseline_wins=2,
            draws=1,
            fine_tuned_win_rate=0.7,
            avg_fine_tuned_score=0.7,
            avg_baseline_score=0.4,
            avg_confidence=0.8,
            consensus_rate=0.9,
        )

        data = result.to_dict()

        assert data["test_id"] == "test-serialize"
        assert data["num_trials"] == 10
        assert data["fine_tuned_win_rate"] == 0.7
        assert "is_significant" in data


# =============================================================================
# EvaluationMetrics Tests
# =============================================================================


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating evaluation metrics."""
        metrics = EvaluationMetrics(
            model_id="model-123",
            elo_rating=1500,
            win_rate=0.65,
            avg_score=0.72,
            calibration_score=0.85,
            consensus_contribution=0.78,
            domain_scores={"security": 1550, "architecture": 1480},
            total_debates=100,
        )

        assert metrics.model_id == "model-123"
        assert metrics.elo_rating == 1500
        assert metrics.domain_scores["security"] == 1550

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = EvaluationMetrics(
            model_id="model",
            elo_rating=1500,
            win_rate=0.5,
            avg_score=0.5,
            calibration_score=0.5,
            consensus_contribution=0.5,
        )

        data = metrics.to_dict()

        assert data["model_id"] == "model"
        assert data["elo_rating"] == 1500
        assert "domain_scores" in data


# =============================================================================
# TinkerEvaluator Tests
# =============================================================================


class TestTinkerEvaluator:
    """Test TinkerEvaluator."""

    def test_generate_test_id(self, evaluator):
        """Test test ID generation."""
        id1 = evaluator._generate_test_id()
        id2 = evaluator._generate_test_id()

        assert id1.startswith("test-")
        assert id2.startswith("test-")
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_a_b_test(self, evaluator, mock_debate_result):
        """Test running A/B test."""
        ft_agent = MagicMock()
        ft_agent.name = "fine-tuned"

        bl_agent = MagicMock()
        bl_agent.name = "baseline"

        with patch.object(evaluator, "_run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_debate_result

            result = await evaluator.a_b_test(
                tasks=["Task 1"],
                fine_tuned_agent=ft_agent,
                baseline_agent=bl_agent,
                num_trials=3,
                save_results=False,
            )

            assert result.num_trials == 3
            assert result.fine_tuned_agent == "fine-tuned"
            assert result.baseline_agent == "baseline"
            assert mock_run.call_count == 3

    def test_save_and_load_result(self, evaluator, temp_dir):
        """Test saving and loading results."""
        result = ABTestResult(
            test_id="test-save-load",
            fine_tuned_agent="ft",
            baseline_agent="bl",
            num_trials=10,
            fine_tuned_wins=6,
            baseline_wins=3,
            draws=1,
            fine_tuned_win_rate=0.6,
            avg_fine_tuned_score=0.65,
            avg_baseline_score=0.45,
            avg_confidence=0.8,
            consensus_rate=0.9,
            trials=[{"trial": 1, "winner": "fine_tuned"}],
        )

        evaluator._save_result(result)

        loaded = evaluator.load_result("test-save-load")

        assert loaded is not None
        assert loaded.test_id == result.test_id
        assert loaded.fine_tuned_wins == result.fine_tuned_wins

    def test_load_nonexistent_result(self, evaluator):
        """Test loading nonexistent result."""
        result = evaluator.load_result("nonexistent")
        assert result is None

    def test_list_results(self, evaluator, temp_dir):
        """Test listing saved results."""
        # Create some results
        for i in range(3):
            result = ABTestResult(
                test_id=f"test-list-{i:04d}",
                fine_tuned_agent="ft",
                baseline_agent="bl",
                num_trials=10,
                fine_tuned_wins=5,
                baseline_wins=4,
                draws=1,
                fine_tuned_win_rate=0.5,
                avg_fine_tuned_score=0.5,
                avg_baseline_score=0.5,
                avg_confidence=0.75,
                consensus_rate=0.8,
            )
            evaluator._save_result(result)

        results = evaluator.list_results(limit=10)

        assert len(results) == 3
        # Check structure
        for r in results:
            assert "test_id" in r
            assert "fine_tuned_win_rate" in r

    def test_get_model_metrics(self, evaluator):
        """Test getting model metrics."""
        # Need to mock EloSystem
        from aragora.ranking.elo import AgentRating

        mock_rating = MagicMock(spec=AgentRating)
        mock_rating.elo = 1500
        mock_rating.win_rate = 0.55
        mock_rating.calibration_score = 0.8
        mock_rating.domain_elos = {"security": 1520}
        mock_rating.games_played = 50

        evaluator.elo.get_rating = MagicMock(return_value=mock_rating)

        metrics = evaluator.get_model_metrics("test-agent")

        assert metrics.model_id == "test-agent"
        assert metrics.elo_rating == 1500
        assert metrics.win_rate == 0.55


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Integration tests for training system."""

    def test_scheduler_job_lifecycle(self, scheduler):
        """Test full job lifecycle."""
        # Create job
        job = TrainingJob(
            job_id="lifecycle-test",
            job_type=JobType.SFT,
            model="model",
            status=JobStatus.PENDING,
        )
        scheduler._jobs["lifecycle-test"] = job

        # Update status
        job.status = JobStatus.PREPARING
        job.started_at = datetime.now().isoformat()

        # Complete
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now().isoformat()
        job.model_id = "output-model-123"

        # Verify
        retrieved = scheduler.get_job("lifecycle-test")
        assert retrieved.status == JobStatus.COMPLETED
        assert retrieved.model_id == "output-model-123"

    @pytest.mark.asyncio
    async def test_evaluator_full_workflow(self, evaluator, mock_debate_result):
        """Test full evaluation workflow."""
        ft_agent = MagicMock()
        ft_agent.name = "fine-tuned"

        bl_agent = MagicMock()
        bl_agent.name = "baseline"

        with patch.object(evaluator, "_run_debate", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_debate_result

            # Run A/B test
            result = await evaluator.a_b_test(
                tasks=["Task 1", "Task 2"],
                fine_tuned_agent=ft_agent,
                baseline_agent=bl_agent,
                num_trials=2,
                save_results=True,
            )

            # Load back
            loaded = evaluator.load_result(result.test_id)
            assert loaded is not None
            assert loaded.test_id == result.test_id

            # List results
            results = evaluator.list_results()
            assert any(r["test_id"] == result.test_id for r in results)

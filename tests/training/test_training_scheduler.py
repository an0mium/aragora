"""Tests for TrainingScheduler module.

Deep tests for JobType, JobStatus, TrainingJob, SchedulerConfig, and TrainingScheduler.
"""

import asyncio
import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.training.training_scheduler import (
    JobType,
    JobStatus,
    TrainingJob,
    SchedulerConfig,
    TrainingScheduler,
)


# =============================================================================
# JobType Enum Tests
# =============================================================================


class TestJobType:
    """Test JobType enum."""

    def test_sft_value(self):
        """Test SFT value."""
        assert JobType.SFT.value == "sft"

    def test_dpo_value(self):
        """Test DPO value."""
        assert JobType.DPO.value == "dpo"

    def test_gauntlet_value(self):
        """Test GAUNTLET value."""
        assert JobType.GAUNTLET.value == "gauntlet"

    def test_combined_value(self):
        """Test COMBINED value."""
        assert JobType.COMBINED.value == "combined"

    def test_all_values(self):
        """Test all enum values."""
        values = [j.value for j in JobType]
        assert "sft" in values
        assert "dpo" in values
        assert "gauntlet" in values
        assert "combined" in values

    def test_from_string(self):
        """Test creating from string."""
        assert JobType("sft") == JobType.SFT
        assert JobType("dpo") == JobType.DPO


# =============================================================================
# JobStatus Enum Tests
# =============================================================================


class TestJobStatus:
    """Test JobStatus enum."""

    def test_pending_value(self):
        """Test PENDING value."""
        assert JobStatus.PENDING.value == "pending"

    def test_preparing_value(self):
        """Test PREPARING value."""
        assert JobStatus.PREPARING.value == "preparing"

    def test_submitted_value(self):
        """Test SUBMITTED value."""
        assert JobStatus.SUBMITTED.value == "submitted"

    def test_running_value(self):
        """Test RUNNING value."""
        assert JobStatus.RUNNING.value == "running"

    def test_completed_value(self):
        """Test COMPLETED value."""
        assert JobStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        """Test FAILED value."""
        assert JobStatus.FAILED.value == "failed"

    def test_cancelled_value(self):
        """Test CANCELLED value."""
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_all_statuses(self):
        """Test all status values exist."""
        statuses = [s.value for s in JobStatus]
        assert len(statuses) == 7

    def test_from_string(self):
        """Test creating from string."""
        assert JobStatus("pending") == JobStatus.PENDING
        assert JobStatus("completed") == JobStatus.COMPLETED


# =============================================================================
# TrainingJob Tests
# =============================================================================


class TestTrainingJobCreation:
    """Test TrainingJob creation."""

    def test_basic_creation(self):
        """Test basic job creation."""
        job = TrainingJob(
            job_id="job-001",
            job_type=JobType.SFT,
            model="llama-3",
        )
        assert job.job_id == "job-001"
        assert job.job_type == JobType.SFT
        assert job.model == "llama-3"

    def test_default_status(self):
        """Test default status is PENDING."""
        job = TrainingJob(
            job_id="job",
            job_type=JobType.DPO,
            model="model",
        )
        assert job.status == JobStatus.PENDING

    def test_timestamps_default(self):
        """Test timestamp defaults."""
        job = TrainingJob(
            job_id="job",
            job_type=JobType.SFT,
            model="model",
        )
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None

    def test_optional_fields_default(self):
        """Test optional fields default to None."""
        job = TrainingJob(
            job_id="job",
            job_type=JobType.SFT,
            model="model",
        )
        assert job.tinker_job_id is None
        assert job.model_id is None
        assert job.result is None
        assert job.error is None

    def test_config_default_empty(self):
        """Test config defaults to empty dict."""
        job = TrainingJob(
            job_id="job",
            job_type=JobType.SFT,
            model="model",
        )
        assert job.config == {}


class TestTrainingJobSerialization:
    """Test TrainingJob serialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict."""
        job = TrainingJob(
            job_id="job-123",
            job_type=JobType.SFT,
            model="llama-3.3-70b",
        )
        d = job.to_dict()

        assert d["job_id"] == "job-123"
        assert d["job_type"] == "sft"
        assert d["model"] == "llama-3.3-70b"
        assert d["status"] == "pending"

    def test_to_dict_all_fields(self):
        """Test to_dict includes all fields."""
        job = TrainingJob(
            job_id="job-full",
            job_type=JobType.COMBINED,
            model="model-x",
            status=JobStatus.COMPLETED,
            tinker_job_id="tinker-123",
            model_id="output-model",
            config={"adapter": "test"},
            error=None,
        )
        job.started_at = "2024-01-01T00:00:00"
        job.completed_at = "2024-01-01T01:00:00"

        d = job.to_dict()

        assert "job_id" in d
        assert "job_type" in d
        assert "model" in d
        assert "status" in d
        assert "created_at" in d
        assert "started_at" in d
        assert "completed_at" in d
        assert "tinker_job_id" in d
        assert "model_id" in d
        assert "config" in d
        assert "error" in d


# =============================================================================
# SchedulerConfig Tests
# =============================================================================


class TestSchedulerConfig:
    """Test SchedulerConfig."""

    def test_default_data_dir(self):
        """Test default data directory."""
        config = SchedulerConfig()
        assert config.data_dir == Path("training_data")

    def test_default_checkpoint_dir(self):
        """Test default checkpoint directory."""
        config = SchedulerConfig()
        assert config.checkpoint_dir == Path("checkpoints")

    def test_default_max_concurrent(self):
        """Test default max concurrent jobs."""
        config = SchedulerConfig()
        assert config.max_concurrent_jobs == 1

    def test_default_sft_min_confidence(self):
        """Test default SFT min confidence."""
        config = SchedulerConfig()
        assert config.sft_min_confidence == 0.7

    def test_default_dpo_min_elo(self):
        """Test default DPO min ELO difference."""
        config = SchedulerConfig()
        assert config.dpo_min_elo_difference == 50.0

    def test_default_gauntlet_min_robustness(self):
        """Test default gauntlet min robustness."""
        config = SchedulerConfig()
        assert config.gauntlet_min_robustness == 0.3

    def test_default_export_limit(self):
        """Test default export limit."""
        config = SchedulerConfig()
        assert config.export_limit == 1000

    def test_default_replay_ratio(self):
        """Test default replay data ratio."""
        config = SchedulerConfig()
        assert config.replay_data_ratio == 0.2

    def test_custom_values(self, tmp_path):
        """Test custom configuration values."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
            max_concurrent_jobs=3,
            sft_min_confidence=0.8,
            dpo_min_elo_difference=100.0,
            export_limit=500,
        )
        assert config.data_dir == tmp_path / "data"
        assert config.checkpoint_dir == tmp_path / "ckpt"
        assert config.max_concurrent_jobs == 3
        assert config.sft_min_confidence == 0.8
        assert config.dpo_min_elo_difference == 100.0
        assert config.export_limit == 500


# =============================================================================
# TrainingScheduler Init Tests
# =============================================================================


class TestTrainingSchedulerInit:
    """Test TrainingScheduler initialization."""

    def test_default_init(self, tmp_path):
        """Test default initialization."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        assert scheduler.config == config
        assert scheduler._jobs == {}
        assert scheduler._job_counter == 0

    def test_creates_directories(self, tmp_path):
        """Test directories are created."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        assert (tmp_path / "data").exists()
        assert (tmp_path / "ckpt").exists()

    def test_lazy_client(self, tmp_path):
        """Test client is lazily loaded."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)
        assert scheduler._client is None


# =============================================================================
# TrainingScheduler Job ID Generation Tests
# =============================================================================


class TestTrainingSchedulerJobId:
    """Test job ID generation."""

    def test_generate_job_id_format(self, tmp_path):
        """Test job ID format."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job_id = scheduler._generate_job_id()

        assert job_id.startswith("job-")
        # Format: job-YYYYMMDD-HHMMSS-0001

    def test_generate_job_id_increments(self, tmp_path):
        """Test job ID counter increments."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        id1 = scheduler._generate_job_id()
        id2 = scheduler._generate_job_id()

        assert id1 != id2
        assert scheduler._job_counter == 2

    def test_generate_job_id_unique(self, tmp_path):
        """Test job IDs are unique."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        ids = [scheduler._generate_job_id() for _ in range(10)]
        assert len(ids) == len(set(ids))


# =============================================================================
# TrainingScheduler Schedule SFT Tests
# =============================================================================


class TestTrainingSchedulerScheduleSFT:
    """Test SFT job scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_sft_creates_job(self, tmp_path):
        """Test scheduling SFT creates a job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="llama-3")

        assert job.job_type == JobType.SFT
        assert job.model == "llama-3"
        assert job.job_id in scheduler._jobs

    @pytest.mark.asyncio
    async def test_schedule_sft_uses_config_defaults(self, tmp_path):
        """Test SFT uses config defaults."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
            sft_min_confidence=0.75,
            export_limit=500,
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="model")

        assert job.config["min_confidence"] == 0.75
        assert job.config["limit"] == 500

    @pytest.mark.asyncio
    async def test_schedule_sft_custom_params(self, tmp_path):
        """Test SFT with custom parameters."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(
                model="custom-model",
                adapter_name="custom-adapter",
                min_confidence=0.9,
                limit=100,
            )

        assert job.config["adapter_name"] == "custom-adapter"
        assert job.config["min_confidence"] == 0.9
        assert job.config["limit"] == 100


# =============================================================================
# TrainingScheduler Schedule DPO Tests
# =============================================================================


class TestTrainingSchedulerScheduleDPO:
    """Test DPO job scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_dpo_creates_job(self, tmp_path):
        """Test scheduling DPO creates a job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            job = await scheduler.schedule_dpo(model="model")

        assert job.job_type == JobType.DPO
        assert job.job_id in scheduler._jobs

    @pytest.mark.asyncio
    async def test_schedule_dpo_with_beta(self, tmp_path):
        """Test DPO with custom beta."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            job = await scheduler.schedule_dpo(model="model", beta=0.2)

        assert job.config["beta"] == 0.2


# =============================================================================
# TrainingScheduler Schedule Combined Tests
# =============================================================================


class TestTrainingSchedulerScheduleCombined:
    """Test combined job scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_combined_creates_job(self, tmp_path):
        """Test scheduling combined creates a job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_combined_job", new_callable=AsyncMock):
            job = await scheduler.schedule_combined(model="model")

        assert job.job_type == JobType.COMBINED
        assert job.job_id in scheduler._jobs

    @pytest.mark.asyncio
    async def test_schedule_combined_limits(self, tmp_path):
        """Test combined job with custom limits."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_combined_job", new_callable=AsyncMock):
            job = await scheduler.schedule_combined(
                model="model",
                sft_limit=200,
                dpo_limit=100,
            )

        assert job.config["sft_limit"] == 200
        assert job.config["dpo_limit"] == 100


# =============================================================================
# TrainingScheduler Job Management Tests
# =============================================================================


class TestTrainingSchedulerJobManagement:
    """Test job management functions."""

    @pytest.mark.asyncio
    async def test_get_job_found(self, tmp_path):
        """Test getting existing job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="model")

        retrieved = scheduler.get_job(job.job_id)
        assert retrieved == job

    def test_get_job_not_found(self, tmp_path):
        """Test getting non-existent job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        retrieved = scheduler.get_job("nonexistent")
        assert retrieved is None


class TestTrainingSchedulerListJobs:
    """Test listing jobs."""

    @pytest.mark.asyncio
    async def test_list_jobs_all(self, tmp_path):
        """Test listing all jobs."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock), \
             patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            await scheduler.schedule_sft(model="m1")
            await scheduler.schedule_dpo(model="m2")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_list_jobs_by_status(self, tmp_path):
        """Test listing jobs by status."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")

        # Manually change status
        job.status = JobStatus.COMPLETED

        pending = scheduler.list_jobs(status=JobStatus.PENDING)
        completed = scheduler.list_jobs(status=JobStatus.COMPLETED)

        assert len(pending) == 0
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_list_jobs_by_type(self, tmp_path):
        """Test listing jobs by type."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock), \
             patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            await scheduler.schedule_sft(model="m1")
            await scheduler.schedule_dpo(model="m2")

        sft_jobs = scheduler.list_jobs(job_type=JobType.SFT)
        dpo_jobs = scheduler.list_jobs(job_type=JobType.DPO)

        assert len(sft_jobs) == 1
        assert len(dpo_jobs) == 1

    @pytest.mark.asyncio
    async def test_list_jobs_limit(self, tmp_path):
        """Test listing jobs with limit."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            for _ in range(10):
                await scheduler.schedule_sft(model="m")

        jobs = scheduler.list_jobs(limit=5)
        assert len(jobs) == 5


# =============================================================================
# TrainingScheduler Cancel Job Tests
# =============================================================================


class TestTrainingSchedulerCancelJob:
    """Test job cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, tmp_path):
        """Test cancelling pending job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")

        result = scheduler.cancel_job(job.job_id)

        assert result is True
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_preparing_job(self, tmp_path):
        """Test cancelling preparing job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")
            job.status = JobStatus.PREPARING

        result = scheduler.cancel_job(job.job_id)

        assert result is True
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_running_job_fails(self, tmp_path):
        """Test cancelling running job fails."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")
            job.status = JobStatus.RUNNING

        result = scheduler.cancel_job(job.job_id)

        assert result is False
        assert job.status == JobStatus.RUNNING

    def test_cancel_nonexistent_job(self, tmp_path):
        """Test cancelling non-existent job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        result = scheduler.cancel_job("nonexistent")
        assert result is False


# =============================================================================
# TrainingScheduler Persistence Tests
# =============================================================================


class TestTrainingSchedulerPersistence:
    """Test state persistence."""

    @pytest.mark.asyncio
    async def test_save_state(self, tmp_path):
        """Test saving state."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            await scheduler.schedule_sft(model="m1")
            await scheduler.schedule_sft(model="m2")

        state_file = tmp_path / "state.json"
        scheduler.save_state(state_file)

        assert state_file.exists()

        with open(state_file) as f:
            state = json.load(f)

        assert len(state["jobs"]) == 2
        assert state["job_counter"] == 2

    def test_load_state(self, tmp_path):
        """Test loading state."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        # Create state file
        state = {
            "jobs": [
                {
                    "job_id": "job-001",
                    "job_type": "sft",
                    "model": "model-1",
                    "status": "completed",
                    "config": {},
                }
            ],
            "job_counter": 1,
        }
        state_file = tmp_path / "state.json"
        with open(state_file, "w") as f:
            json.dump(state, f)

        scheduler.load_state(state_file)

        assert scheduler._job_counter == 1
        assert "job-001" in scheduler._jobs
        assert scheduler._jobs["job-001"].status == JobStatus.COMPLETED

    def test_load_state_missing_file(self, tmp_path):
        """Test loading from missing file."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        # Should not raise
        scheduler.load_state(tmp_path / "nonexistent.json")

        assert scheduler._job_counter == 0
        assert scheduler._jobs == {}


# =============================================================================
# TrainingScheduler Wait for Job Tests
# =============================================================================


class TestTrainingSchedulerWaitForJob:
    """Test waiting for job completion."""

    @pytest.mark.asyncio
    async def test_wait_for_completed_job(self, tmp_path):
        """Test waiting for already completed job."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")
            job.status = JobStatus.COMPLETED

        result = await scheduler.wait_for_job(job.job_id, poll_interval=0.01, timeout=1.0)
        assert result.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_for_nonexistent_job(self, tmp_path):
        """Test waiting for non-existent job raises."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with pytest.raises(ValueError, match="Job not found"):
            await scheduler.wait_for_job("nonexistent")

    @pytest.mark.asyncio
    async def test_wait_for_job_timeout(self, tmp_path):
        """Test waiting times out."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")
            # Keep status as PENDING

        with pytest.raises(TimeoutError):
            await scheduler.wait_for_job(job.job_id, poll_interval=0.01, timeout=0.05)

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

        with (
            patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock),
            patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock),
        ):
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

        with (
            patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock),
            patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock),
        ):
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

    @pytest.mark.asyncio
    async def test_wait_for_failed_job(self, tmp_path):
        """Test waiting for failed job returns immediately."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")
            job.status = JobStatus.FAILED
            job.error = "Test error"

        result = await scheduler.wait_for_job(job.job_id, poll_interval=0.01, timeout=1.0)
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_wait_for_cancelled_job(self, tmp_path):
        """Test waiting for cancelled job returns immediately."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")
            job.status = JobStatus.CANCELLED

        result = await scheduler.wait_for_job(job.job_id, poll_interval=0.01, timeout=1.0)
        assert result.status == JobStatus.CANCELLED


# =============================================================================
# TrainingScheduler Client and Close Tests
# =============================================================================


class TestTrainingSchedulerClient:
    """Test client property and close method."""

    def test_client_lazy_loading(self, tmp_path):
        """Test client is lazily loaded."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        assert scheduler._client is None

        # Access the client property
        client = scheduler.client

        assert client is not None
        assert scheduler._client is not None

    @pytest.mark.asyncio
    async def test_close_with_client(self, tmp_path):
        """Test close when client exists."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        # Create a mock client
        mock_client = AsyncMock()
        scheduler._client = mock_client

        await scheduler.close()

        mock_client.close.assert_called_once()
        assert scheduler._client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, tmp_path):
        """Test close when no client exists."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        # Should not raise
        await scheduler.close()

        assert scheduler._client is None


# =============================================================================
# TrainingScheduler Run SFT Job Tests
# =============================================================================


class TestRunSFTJob:
    """Test _run_sft_job execution."""

    @pytest.mark.asyncio
    async def test_run_sft_job_successful_completion(self, tmp_path):
        """Test SFT job completes successfully."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-sft-001",
            job_type=JobType.SFT,
            model="llama-3",
            config={
                "adapter_name": "test-adapter",
                "min_confidence": 0.7,
                "limit": 100,
            },
        )
        scheduler._jobs[job.job_id] = job

        # Mock exporter
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [{"instruction": "test", "response": "response"}]

        # Mock client training result
        mock_result = TrainingResult(
            job_id="tinker-job-001",
            state=TrainingState.COMPLETED,
            model_id="model-output-001",
            final_loss=0.1,
            total_steps=100,
            training_time_seconds=60.0,
            checkpoint_path="/checkpoints/model",
        )
        mock_client = AsyncMock()
        mock_client.train_sft.return_value = mock_result

        with patch("aragora.training.training_scheduler.SFTExporter", return_value=mock_exporter):
            scheduler._client = mock_client
            await scheduler._run_sft_job(job)

        assert job.status == JobStatus.COMPLETED
        assert job.model_id == "model-output-001"
        assert job.tinker_job_id == "tinker-job-001"
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_sft_job_running_state(self, tmp_path):
        """Test SFT job in running state."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-sft-002",
            job_type=JobType.SFT,
            model="llama-3",
            config={
                "adapter_name": "test-adapter",
                "min_confidence": 0.7,
                "limit": 100,
            },
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [{"instruction": "test", "response": "response"}]

        mock_result = TrainingResult(
            job_id="tinker-job-002",
            state=TrainingState.RUNNING,
            model_id=None,
            final_loss=None,
            total_steps=50,
            training_time_seconds=30.0,
            checkpoint_path=None,
        )
        mock_client = AsyncMock()
        mock_client.train_sft.return_value = mock_result

        with patch("aragora.training.training_scheduler.SFTExporter", return_value=mock_exporter):
            scheduler._client = mock_client
            await scheduler._run_sft_job(job)

        assert job.status == JobStatus.RUNNING
        assert job.tinker_job_id == "tinker-job-002"

    @pytest.mark.asyncio
    async def test_run_sft_job_failed_state(self, tmp_path):
        """Test SFT job fails from Tinker API."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-sft-003",
            job_type=JobType.SFT,
            model="llama-3",
            config={
                "adapter_name": "test-adapter",
                "min_confidence": 0.7,
                "limit": 100,
            },
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [{"instruction": "test", "response": "response"}]

        mock_result = TrainingResult(
            job_id="tinker-job-003",
            state=TrainingState.FAILED,
            model_id=None,
            final_loss=None,
            total_steps=10,
            training_time_seconds=5.0,
            checkpoint_path=None,
            error_message="Training failed: OOM",
        )
        mock_client = AsyncMock()
        mock_client.train_sft.return_value = mock_result

        with patch("aragora.training.training_scheduler.SFTExporter", return_value=mock_exporter):
            scheduler._client = mock_client
            await scheduler._run_sft_job(job)

        assert job.status == JobStatus.FAILED
        assert job.error == "Training failed: OOM"
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_sft_job_no_data(self, tmp_path):
        """Test SFT job fails when no data exported."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-sft-004",
            job_type=JobType.SFT,
            model="llama-3",
            config={"min_confidence": 0.7, "limit": 100},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []  # No data

        with patch("aragora.training.training_scheduler.SFTExporter", return_value=mock_exporter):
            await scheduler._run_sft_job(job)

        assert job.status == JobStatus.FAILED
        assert "No training data exported" in job.error

    @pytest.mark.asyncio
    async def test_run_sft_job_exception(self, tmp_path):
        """Test SFT job handles exceptions."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-sft-005",
            job_type=JobType.SFT,
            model="llama-3",
            config={"min_confidence": 0.7, "limit": 100},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.side_effect = OSError("Disk full")

        with patch("aragora.training.training_scheduler.SFTExporter", return_value=mock_exporter):
            await scheduler._run_sft_job(job)

        assert job.status == JobStatus.FAILED
        assert "Disk full" in job.error

    @pytest.mark.asyncio
    async def test_run_sft_job_io_error(self, tmp_path):
        """Test SFT job handles IOError."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-sft-006",
            job_type=JobType.SFT,
            model="llama-3",
            config={"min_confidence": 0.7, "limit": 100},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.side_effect = OSError("Cannot write")

        with patch("aragora.training.training_scheduler.SFTExporter", return_value=mock_exporter):
            await scheduler._run_sft_job(job)

        assert job.status == JobStatus.FAILED
        assert "Cannot write" in job.error

    @pytest.mark.asyncio
    async def test_run_sft_job_runtime_error(self, tmp_path):
        """Test SFT job handles RuntimeError."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-sft-007",
            job_type=JobType.SFT,
            model="llama-3",
            config={"min_confidence": 0.7, "limit": 100},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.side_effect = RuntimeError("Runtime issue")

        with patch("aragora.training.training_scheduler.SFTExporter", return_value=mock_exporter):
            await scheduler._run_sft_job(job)

        assert job.status == JobStatus.FAILED
        assert "Runtime issue" in job.error

    @pytest.mark.asyncio
    async def test_run_sft_job_creates_data_file(self, tmp_path):
        """Test SFT job creates data file."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-sft-008",
            job_type=JobType.SFT,
            model="llama-3",
            config={"min_confidence": 0.7, "limit": 100},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"instruction": "test1", "response": "response1"},
            {"instruction": "test2", "response": "response2"},
        ]

        mock_result = TrainingResult(
            job_id="tinker-job",
            state=TrainingState.COMPLETED,
            model_id="model-001",
            final_loss=0.1,
            total_steps=100,
            training_time_seconds=60.0,
            checkpoint_path=None,
        )
        mock_client = AsyncMock()
        mock_client.train_sft.return_value = mock_result

        with patch("aragora.training.training_scheduler.SFTExporter", return_value=mock_exporter):
            scheduler._client = mock_client
            await scheduler._run_sft_job(job)

        # Check data file was created
        data_file = tmp_path / "data" / f"{job.job_id}_sft.jsonl"
        assert data_file.exists()

        # Check content
        with open(data_file) as f:
            lines = f.readlines()
        assert len(lines) == 2


# =============================================================================
# TrainingScheduler Run DPO Job Tests
# =============================================================================


class TestRunDPOJob:
    """Test _run_dpo_job execution."""

    @pytest.mark.asyncio
    async def test_run_dpo_job_successful_completion(self, tmp_path):
        """Test DPO job completes successfully."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-dpo-001",
            job_type=JobType.DPO,
            model="llama-3",
            config={
                "adapter_name": "test-adapter",
                "min_elo_difference": 50.0,
                "limit": 100,
                "beta": 0.1,
            },
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"prompt": "test", "chosen": "good", "rejected": "bad"}
        ]

        mock_result = TrainingResult(
            job_id="tinker-dpo-001",
            state=TrainingState.COMPLETED,
            model_id="dpo-model-001",
            final_loss=0.05,
            total_steps=50,
            training_time_seconds=30.0,
            checkpoint_path="/checkpoints/dpo",
        )
        mock_client = AsyncMock()
        mock_client.train_dpo.return_value = mock_result

        with patch("aragora.training.training_scheduler.DPOExporter", return_value=mock_exporter):
            scheduler._client = mock_client
            await scheduler._run_dpo_job(job)

        assert job.status == JobStatus.COMPLETED
        assert job.model_id == "dpo-model-001"
        assert job.tinker_job_id == "tinker-dpo-001"

    @pytest.mark.asyncio
    async def test_run_dpo_job_failed_state(self, tmp_path):
        """Test DPO job fails from Tinker API."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-dpo-002",
            job_type=JobType.DPO,
            model="llama-3",
            config={"min_elo_difference": 50.0, "limit": 100, "beta": 0.1},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"prompt": "test", "chosen": "good", "rejected": "bad"}
        ]

        mock_result = TrainingResult(
            job_id="tinker-dpo-002",
            state=TrainingState.FAILED,
            model_id=None,
            final_loss=None,
            total_steps=5,
            training_time_seconds=2.0,
            checkpoint_path=None,
            error_message="DPO training failed",
        )
        mock_client = AsyncMock()
        mock_client.train_dpo.return_value = mock_result

        with patch("aragora.training.training_scheduler.DPOExporter", return_value=mock_exporter):
            scheduler._client = mock_client
            await scheduler._run_dpo_job(job)

        assert job.status == JobStatus.FAILED
        assert job.error == "DPO training failed"

    @pytest.mark.asyncio
    async def test_run_dpo_job_no_data(self, tmp_path):
        """Test DPO job fails when no data exported."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-dpo-003",
            job_type=JobType.DPO,
            model="llama-3",
            config={"min_elo_difference": 50.0, "limit": 100, "beta": 0.1},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = []

        with patch("aragora.training.training_scheduler.DPOExporter", return_value=mock_exporter):
            await scheduler._run_dpo_job(job)

        assert job.status == JobStatus.FAILED
        assert "No preference data exported" in job.error

    @pytest.mark.asyncio
    async def test_run_dpo_job_exception(self, tmp_path):
        """Test DPO job handles exceptions."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-dpo-004",
            job_type=JobType.DPO,
            model="llama-3",
            config={"min_elo_difference": 50.0, "limit": 100, "beta": 0.1},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.side_effect = ValueError("Invalid data format")

        with patch("aragora.training.training_scheduler.DPOExporter", return_value=mock_exporter):
            await scheduler._run_dpo_job(job)

        assert job.status == JobStatus.FAILED
        assert "Invalid data format" in job.error

    @pytest.mark.asyncio
    async def test_run_dpo_job_creates_data_file(self, tmp_path):
        """Test DPO job creates data file."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-dpo-005",
            job_type=JobType.DPO,
            model="llama-3",
            config={"min_elo_difference": 50.0, "limit": 100, "beta": 0.1},
        )
        scheduler._jobs[job.job_id] = job

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [
            {"prompt": "p1", "chosen": "c1", "rejected": "r1"},
            {"prompt": "p2", "chosen": "c2", "rejected": "r2"},
        ]

        mock_result = TrainingResult(
            job_id="tinker-dpo",
            state=TrainingState.COMPLETED,
            model_id="model-001",
            final_loss=0.05,
            total_steps=50,
            training_time_seconds=30.0,
            checkpoint_path=None,
        )
        mock_client = AsyncMock()
        mock_client.train_dpo.return_value = mock_result

        with patch("aragora.training.training_scheduler.DPOExporter", return_value=mock_exporter):
            scheduler._client = mock_client
            await scheduler._run_dpo_job(job)

        data_file = tmp_path / "data" / f"{job.job_id}_dpo.jsonl"
        assert data_file.exists()


# =============================================================================
# TrainingScheduler Run Combined Job Tests
# =============================================================================


class TestRunCombinedJob:
    """Test _run_combined_job execution."""

    @pytest.mark.asyncio
    async def test_run_combined_job_successful(self, tmp_path):
        """Test combined job completes successfully."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-combined-001",
            job_type=JobType.COMBINED,
            model="llama-3",
            config={
                "adapter_name": "combined-adapter",
                "sft_limit": 100,
                "dpo_limit": 50,
            },
        )
        scheduler._jobs[job.job_id] = job

        mock_sft_exporter = MagicMock()
        mock_sft_exporter.export.return_value = [{"instruction": "test", "response": "response"}]

        mock_dpo_exporter = MagicMock()
        mock_dpo_exporter.export.return_value = [{"prompt": "p", "chosen": "c", "rejected": "r"}]

        sft_result = TrainingResult(
            job_id="sft-job",
            state=TrainingState.COMPLETED,
            model_id="sft-model-001",
            final_loss=0.1,
            total_steps=100,
            training_time_seconds=60.0,
            checkpoint_path=None,
        )

        dpo_result = TrainingResult(
            job_id="dpo-job",
            state=TrainingState.COMPLETED,
            model_id="dpo-model-001",
            final_loss=0.05,
            total_steps=50,
            training_time_seconds=30.0,
            checkpoint_path=None,
        )

        mock_client = AsyncMock()
        mock_client.train_sft.return_value = sft_result
        mock_client.train_dpo.return_value = dpo_result

        with (
            patch(
                "aragora.training.training_scheduler.SFTExporter", return_value=mock_sft_exporter
            ),
            patch(
                "aragora.training.training_scheduler.DPOExporter", return_value=mock_dpo_exporter
            ),
        ):
            scheduler._client = mock_client
            await scheduler._run_combined_job(job)

        assert job.status == JobStatus.COMPLETED
        assert job.model_id == "dpo-model-001"
        mock_client.train_dpo.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_combined_job_sft_only(self, tmp_path):
        """Test combined job falls back to SFT when no DPO data."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-combined-002",
            job_type=JobType.COMBINED,
            model="llama-3",
            config={"adapter_name": "combined-adapter"},
        )
        scheduler._jobs[job.job_id] = job

        mock_sft_exporter = MagicMock()
        mock_sft_exporter.export.return_value = [{"instruction": "test", "response": "response"}]

        mock_dpo_exporter = MagicMock()
        mock_dpo_exporter.export.return_value = []  # No DPO data

        sft_result = TrainingResult(
            job_id="sft-job",
            state=TrainingState.COMPLETED,
            model_id="sft-model-only",
            final_loss=0.1,
            total_steps=100,
            training_time_seconds=60.0,
            checkpoint_path=None,
        )

        mock_client = AsyncMock()
        mock_client.train_sft.return_value = sft_result

        with (
            patch(
                "aragora.training.training_scheduler.SFTExporter", return_value=mock_sft_exporter
            ),
            patch(
                "aragora.training.training_scheduler.DPOExporter", return_value=mock_dpo_exporter
            ),
        ):
            scheduler._client = mock_client
            await scheduler._run_combined_job(job)

        assert job.status == JobStatus.COMPLETED
        assert job.model_id == "sft-model-only"
        mock_client.train_dpo.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_combined_job_sft_fails(self, tmp_path):
        """Test combined job fails when SFT phase fails."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-combined-003",
            job_type=JobType.COMBINED,
            model="llama-3",
            config={"adapter_name": "combined-adapter"},
        )
        scheduler._jobs[job.job_id] = job

        mock_sft_exporter = MagicMock()
        mock_sft_exporter.export.return_value = [{"instruction": "test", "response": "response"}]

        sft_result = TrainingResult(
            job_id="sft-job",
            state=TrainingState.FAILED,
            model_id=None,
            final_loss=None,
            total_steps=10,
            training_time_seconds=5.0,
            checkpoint_path=None,
            error_message="SFT phase failed",
        )

        mock_client = AsyncMock()
        mock_client.train_sft.return_value = sft_result

        with patch(
            "aragora.training.training_scheduler.SFTExporter", return_value=mock_sft_exporter
        ):
            scheduler._client = mock_client
            await scheduler._run_combined_job(job)

        assert job.status == JobStatus.FAILED
        assert "SFT phase failed" in job.error

    @pytest.mark.asyncio
    async def test_run_combined_job_dpo_fails(self, tmp_path):
        """Test combined job fails when DPO phase fails."""
        from aragora.training.tinker_client import TrainingResult, TrainingState

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-combined-004",
            job_type=JobType.COMBINED,
            model="llama-3",
            config={"adapter_name": "combined-adapter"},
        )
        scheduler._jobs[job.job_id] = job

        mock_sft_exporter = MagicMock()
        mock_sft_exporter.export.return_value = [{"instruction": "test", "response": "response"}]

        mock_dpo_exporter = MagicMock()
        mock_dpo_exporter.export.return_value = [{"prompt": "p", "chosen": "c", "rejected": "r"}]

        sft_result = TrainingResult(
            job_id="sft-job",
            state=TrainingState.COMPLETED,
            model_id="sft-model",
            final_loss=0.1,
            total_steps=100,
            training_time_seconds=60.0,
            checkpoint_path=None,
        )

        dpo_result = TrainingResult(
            job_id="dpo-job",
            state=TrainingState.FAILED,
            model_id=None,
            final_loss=None,
            total_steps=5,
            training_time_seconds=2.0,
            checkpoint_path=None,
            error_message="DPO failed",
        )

        mock_client = AsyncMock()
        mock_client.train_sft.return_value = sft_result
        mock_client.train_dpo.return_value = dpo_result

        with (
            patch(
                "aragora.training.training_scheduler.SFTExporter", return_value=mock_sft_exporter
            ),
            patch(
                "aragora.training.training_scheduler.DPOExporter", return_value=mock_dpo_exporter
            ),
        ):
            scheduler._client = mock_client
            await scheduler._run_combined_job(job)

        assert job.status == JobStatus.FAILED
        assert job.error == "DPO failed"

    @pytest.mark.asyncio
    async def test_run_combined_job_no_sft_data(self, tmp_path):
        """Test combined job fails when no SFT data."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-combined-005",
            job_type=JobType.COMBINED,
            model="llama-3",
            config={"adapter_name": "combined-adapter"},
        )
        scheduler._jobs[job.job_id] = job

        mock_sft_exporter = MagicMock()
        mock_sft_exporter.export.return_value = []

        with patch(
            "aragora.training.training_scheduler.SFTExporter", return_value=mock_sft_exporter
        ):
            await scheduler._run_combined_job(job)

        assert job.status == JobStatus.FAILED
        assert "No SFT training data exported" in job.error

    @pytest.mark.asyncio
    async def test_run_combined_job_exception(self, tmp_path):
        """Test combined job handles exceptions."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        job = TrainingJob(
            job_id="test-combined-006",
            job_type=JobType.COMBINED,
            model="llama-3",
            config={"adapter_name": "combined-adapter"},
        )
        scheduler._jobs[job.job_id] = job

        mock_sft_exporter = MagicMock()
        mock_sft_exporter.export.side_effect = RuntimeError("Unexpected error")

        with patch(
            "aragora.training.training_scheduler.SFTExporter", return_value=mock_sft_exporter
        ):
            await scheduler._run_combined_job(job)

        assert job.status == JobStatus.FAILED
        assert "Unexpected error" in job.error


# =============================================================================
# TrainingScheduler TinkerModel Handling Tests
# =============================================================================


class TestTinkerModelHandling:
    """Test handling of TinkerModel enum."""

    @pytest.mark.asyncio
    async def test_schedule_sft_with_tinker_model_enum(self, tmp_path):
        """Test SFT with TinkerModel enum."""
        from aragora.training.tinker_client import TinkerModel

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model=TinkerModel.LLAMA_3_3_70B)

        assert job.model == "llama-3.3-70b"

    @pytest.mark.asyncio
    async def test_schedule_dpo_with_tinker_model_enum(self, tmp_path):
        """Test DPO with TinkerModel enum."""
        from aragora.training.tinker_client import TinkerModel

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            job = await scheduler.schedule_dpo(model=TinkerModel.QWEN_2_5_72B)

        assert job.model == "qwen-2.5-72b"

    @pytest.mark.asyncio
    async def test_schedule_combined_with_tinker_model_enum(self, tmp_path):
        """Test combined with TinkerModel enum."""
        from aragora.training.tinker_client import TinkerModel

        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_combined_job", new_callable=AsyncMock):
            job = await scheduler.schedule_combined(model=TinkerModel.DEEPSEEK_V3)

        assert job.model == "deepseek-v3"


# =============================================================================
# TrainingScheduler Auto Adapter Name Tests
# =============================================================================


class TestAutoAdapterName:
    """Test automatic adapter name generation."""

    @pytest.mark.asyncio
    async def test_sft_auto_adapter_name(self, tmp_path):
        """Test SFT generates adapter name from job ID."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="llama-3")

        assert job.config["adapter_name"].startswith("aragora-sft-")
        assert job.job_id in job.config["adapter_name"]

    @pytest.mark.asyncio
    async def test_dpo_auto_adapter_name(self, tmp_path):
        """Test DPO generates adapter name from job ID."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            job = await scheduler.schedule_dpo(model="llama-3")

        assert job.config["adapter_name"].startswith("aragora-dpo-")
        assert job.job_id in job.config["adapter_name"]

    @pytest.mark.asyncio
    async def test_combined_auto_adapter_name(self, tmp_path):
        """Test combined generates adapter name from job ID."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_combined_job", new_callable=AsyncMock):
            job = await scheduler.schedule_combined(model="llama-3")

        assert job.config["adapter_name"].startswith("aragora-combined-")
        assert job.job_id in job.config["adapter_name"]


# =============================================================================
# TrainingScheduler State Persistence Full Cycle Tests
# =============================================================================


class TestStatePersistenceFullCycle:
    """Test full save and load cycle."""

    @pytest.mark.asyncio
    async def test_save_and_load_preserves_all_fields(self, tmp_path):
        """Test full save and load cycle preserves all job fields."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler1 = TrainingScheduler(config=config)

        with patch.object(scheduler1, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler1.schedule_sft(model="llama-3")
            job.status = JobStatus.COMPLETED
            job.tinker_job_id = "tinker-123"
            job.model_id = "model-456"
            job.started_at = "2024-01-01T10:00:00"
            job.completed_at = "2024-01-01T11:00:00"

        # Save state
        state_file = tmp_path / "scheduler_state.json"
        scheduler1.save_state(state_file)

        # Create new scheduler and load
        scheduler2 = TrainingScheduler(config=config)
        scheduler2.load_state(state_file)

        # Verify loaded job
        loaded_job = scheduler2.get_job(job.job_id)
        assert loaded_job is not None
        assert loaded_job.status == JobStatus.COMPLETED
        assert loaded_job.tinker_job_id == "tinker-123"
        assert loaded_job.model_id == "model-456"
        assert loaded_job.started_at == "2024-01-01T10:00:00"
        assert loaded_job.completed_at == "2024-01-01T11:00:00"

    @pytest.mark.asyncio
    async def test_save_preserves_job_counter(self, tmp_path):
        """Test save preserves job counter."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler1 = TrainingScheduler(config=config)

        with patch.object(scheduler1, "_run_sft_job", new_callable=AsyncMock):
            await scheduler1.schedule_sft(model="m1")
            await scheduler1.schedule_sft(model="m2")
            await scheduler1.schedule_sft(model="m3")

        state_file = tmp_path / "state.json"
        scheduler1.save_state(state_file)

        scheduler2 = TrainingScheduler(config=config)
        scheduler2.load_state(state_file)

        assert scheduler2._job_counter == 3


# =============================================================================
# TrainingScheduler DPO Min ELO Difference Tests
# =============================================================================


class TestDPOMinEloDifference:
    """Test DPO min ELO difference configuration."""

    @pytest.mark.asyncio
    async def test_dpo_uses_config_min_elo_difference(self, tmp_path):
        """Test DPO uses config min ELO difference."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
            dpo_min_elo_difference=75.0,
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            job = await scheduler.schedule_dpo(model="llama-3")

        assert job.config["min_elo_difference"] == 75.0

    @pytest.mark.asyncio
    async def test_dpo_custom_min_elo_overrides_config(self, tmp_path):
        """Test DPO custom min ELO overrides config."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
            dpo_min_elo_difference=75.0,
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            job = await scheduler.schedule_dpo(model="llama-3", min_elo_difference=100.0)

        assert job.config["min_elo_difference"] == 100.0


# =============================================================================
# TrainingScheduler Job Sorting Tests
# =============================================================================


class TestJobSorting:
    """Test job listing sort order."""

    @pytest.mark.asyncio
    async def test_list_jobs_sorted_by_creation_time(self, tmp_path):
        """Test jobs are sorted by creation time, newest first."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job1 = await scheduler.schedule_sft(model="m1")
            await asyncio.sleep(0.01)
            job2 = await scheduler.schedule_sft(model="m2")
            await asyncio.sleep(0.01)
            job3 = await scheduler.schedule_sft(model="m3")

        jobs = scheduler.list_jobs()

        # Newest first
        assert jobs[0].job_id == job3.job_id
        assert jobs[1].job_id == job2.job_id
        assert jobs[2].job_id == job1.job_id


# =============================================================================
# TrainingScheduler Combined Filtering Tests
# =============================================================================


class TestCombinedFiltering:
    """Test combined filtering by status and type."""

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_status_and_type(self, tmp_path):
        """Test filtering by both status and type."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with (
            patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock),
            patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock),
        ):
            sft_completed = await scheduler.schedule_sft(model="m1")
            sft_completed.status = JobStatus.COMPLETED

            sft_pending = await scheduler.schedule_sft(model="m2")
            # Keep pending

            dpo_completed = await scheduler.schedule_dpo(model="m3")
            dpo_completed.status = JobStatus.COMPLETED

        # Filter SFT completed
        sft_completed_jobs = scheduler.list_jobs(status=JobStatus.COMPLETED, job_type=JobType.SFT)
        assert len(sft_completed_jobs) == 1
        assert sft_completed_jobs[0].job_id == sft_completed.job_id

        # Filter DPO completed
        dpo_completed_jobs = scheduler.list_jobs(status=JobStatus.COMPLETED, job_type=JobType.DPO)
        assert len(dpo_completed_jobs) == 1
        assert dpo_completed_jobs[0].job_id == dpo_completed.job_id


# =============================================================================
# TrainingScheduler Cancel Sets Completed At Tests
# =============================================================================


class TestCancelSetsCompletedAt:
    """Test cancellation sets completed_at timestamp."""

    @pytest.mark.asyncio
    async def test_cancel_sets_completed_at(self, tmp_path):
        """Test cancel sets completed_at timestamp."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(model="m")

        assert job.completed_at is None

        scheduler.cancel_job(job.job_id)

        assert job.completed_at is not None


# =============================================================================
# TrainingScheduler Extra Config Kwargs Tests
# =============================================================================


class TestExtraConfigKwargs:
    """Test passing extra configuration kwargs."""

    @pytest.mark.asyncio
    async def test_sft_passes_extra_kwargs(self, tmp_path):
        """Test SFT passes extra kwargs to config."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_sft_job", new_callable=AsyncMock):
            job = await scheduler.schedule_sft(
                model="llama-3",
                custom_param="custom_value",
                lora_rank=32,
            )

        assert job.config["custom_param"] == "custom_value"
        assert job.config["lora_rank"] == 32

    @pytest.mark.asyncio
    async def test_dpo_passes_extra_kwargs(self, tmp_path):
        """Test DPO passes extra kwargs to config."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_dpo_job", new_callable=AsyncMock):
            job = await scheduler.schedule_dpo(
                model="llama-3",
                custom_dpo_param="dpo_value",
            )

        assert job.config["custom_dpo_param"] == "dpo_value"

    @pytest.mark.asyncio
    async def test_combined_passes_extra_kwargs(self, tmp_path):
        """Test combined passes extra kwargs to config."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        with patch.object(scheduler, "_run_combined_job", new_callable=AsyncMock):
            job = await scheduler.schedule_combined(
                model="llama-3",
                custom_combined="value",
            )

        assert job.config["custom_combined"] == "value"


# =============================================================================
# TrainingScheduler Default Model Tests
# =============================================================================


class TestDefaultModel:
    """Test default model configuration."""

    def test_scheduler_config_default_model(self):
        """Test SchedulerConfig has default model."""
        from aragora.training.tinker_client import TinkerModel

        config = SchedulerConfig()
        assert config.default_model == TinkerModel.LLAMA_3_3_70B.value


# =============================================================================
# TrainingScheduler Load State with Error Field Tests
# =============================================================================


class TestLoadStateWithError:
    """Test loading state with error field."""

    def test_load_state_with_error(self, tmp_path):
        """Test loading state that includes error field."""
        config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        scheduler = TrainingScheduler(config=config)

        state = {
            "jobs": [
                {
                    "job_id": "job-failed",
                    "job_type": "sft",
                    "model": "llama-3",
                    "status": "failed",
                    "config": {},
                    "error": "Training failed due to OOM",
                    "started_at": "2024-01-01T10:00:00",
                    "completed_at": "2024-01-01T10:05:00",
                }
            ],
            "job_counter": 1,
        }

        state_file = tmp_path / "state.json"
        with open(state_file, "w") as f:
            json.dump(state, f)

        scheduler.load_state(state_file)

        job = scheduler.get_job("job-failed")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error == "Training failed due to OOM"


# =============================================================================
# TrainingJob to_dict Result Field Tests
# =============================================================================


class TestTrainingJobResultField:
    """Test TrainingJob result field handling."""

    def test_to_dict_excludes_result(self):
        """Test to_dict does not include result field (it's not serializable)."""
        job = TrainingJob(
            job_id="job-001",
            job_type=JobType.SFT,
            model="llama-3",
        )

        d = job.to_dict()

        # result is not included in to_dict
        assert "result" not in d


# =============================================================================
# TrainingScheduler Tinker Config Tests
# =============================================================================


class TestTinkerConfig:
    """Test Tinker config handling."""

    def test_scheduler_uses_provided_tinker_config(self, tmp_path):
        """Test scheduler uses provided Tinker config."""
        from aragora.training.tinker_client import TinkerConfig

        scheduler_config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )
        tinker_config = TinkerConfig(
            api_key="test-api-key",
            base_url="https://custom.api.com/v1",
        )

        scheduler = TrainingScheduler(
            config=scheduler_config,
            tinker_config=tinker_config,
        )

        assert scheduler.tinker_config == tinker_config
        assert scheduler.tinker_config.api_key == "test-api-key"
        assert scheduler.tinker_config.base_url == "https://custom.api.com/v1"

    def test_scheduler_creates_default_tinker_config(self, tmp_path):
        """Test scheduler creates default Tinker config if not provided."""
        from aragora.training.tinker_client import TinkerConfig

        scheduler_config = SchedulerConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "ckpt",
        )

        scheduler = TrainingScheduler(config=scheduler_config)

        assert scheduler.tinker_config is not None
        assert isinstance(scheduler.tinker_config, TinkerConfig)

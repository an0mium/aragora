"""
Training scheduler for batch Tinker jobs.

Manages training job scheduling, data preparation, and model lifecycle.
Designed for offline/batch training rather than real-time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import asyncio
import json
import logging
import os

from aragora.training.tinker_client import (
    TinkerClient,
    TinkerConfig,
    TinkerModel,
    TrainingResult,
    TrainingState,
)
from aragora.training.exporters import SFTExporter, DPOExporter, GauntletExporter

logger = logging.getLogger(__name__)


class JobType(str, Enum):
    """Type of training job."""

    SFT = "sft"
    DPO = "dpo"
    GAUNTLET = "gauntlet"
    COMBINED = "combined"


class JobStatus(str, Enum):
    """Status of a scheduled job."""

    PENDING = "pending"
    PREPARING = "preparing"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """A scheduled training job."""

    job_id: str
    job_type: JobType
    model: str
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    tinker_job_id: str | None = None
    model_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    result: TrainingResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "model": self.model,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tinker_job_id": self.tinker_job_id,
            "model_id": self.model_id,
            "config": self.config,
            "error": self.error,
        }


@dataclass
class SchedulerConfig:
    """Configuration for training scheduler."""

    data_dir: Path = field(default_factory=lambda: Path("training_data"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    max_concurrent_jobs: int = 1
    default_model: str = TinkerModel.LLAMA_3_3_70B.value
    sft_min_confidence: float = 0.7
    dpo_min_elo_difference: float = 50.0
    gauntlet_min_robustness: float = 0.3
    export_limit: int = 1000
    replay_data_ratio: float = 0.2  # Mix 20% historical data to prevent forgetting


class TrainingScheduler:
    """
    Scheduler for Tinker training jobs.

    Manages the full training pipeline:
    1. Export training data from Aragora stores
    2. Prepare and validate data
    3. Submit jobs to Tinker API
    4. Track job progress
    5. Manage model lifecycle

    Example:
        scheduler = TrainingScheduler()

        # Schedule an SFT training job
        job = await scheduler.schedule_sft(
            model=TinkerModel.LLAMA_3_3_70B,
            adapter_name="aragora-security-v1",
        )

        # Wait for completion
        result = await scheduler.wait_for_job(job.job_id)
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        tinker_config: TinkerConfig | None = None,
    ):
        self.config = config or SchedulerConfig()
        self.tinker_config = tinker_config or TinkerConfig()
        self._client: TinkerClient | None = None
        self._jobs: dict[str, TrainingJob] = {}
        self._job_counter = 0

        # Ensure directories exist
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> TinkerClient:
        """Lazy-load Tinker client."""
        if self._client is None:
            self._client = TinkerClient(self.tinker_config)
        return self._client

    async def close(self) -> None:
        """Close resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        self._job_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"job-{timestamp}-{self._job_counter:04d}"

    async def schedule_sft(
        self,
        model: str | TinkerModel = TinkerModel.LLAMA_3_3_70B,
        adapter_name: str | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
        **kwargs,
    ) -> TrainingJob:
        """
        Schedule an SFT training job.

        Args:
            model: Base model to fine-tune
            adapter_name: Name for the adapter
            min_confidence: Minimum debate confidence for training data
            limit: Maximum training examples
            **kwargs: Additional training config

        Returns:
            TrainingJob with job details
        """
        if isinstance(model, TinkerModel):
            model = model.value

        job_id = self._generate_job_id()
        adapter_name = adapter_name or f"aragora-sft-{job_id}"

        job = TrainingJob(
            job_id=job_id,
            job_type=JobType.SFT,
            model=model,
            config={
                "adapter_name": adapter_name,
                "min_confidence": min_confidence or self.config.sft_min_confidence,
                "limit": limit or self.config.export_limit,
                **kwargs,
            },
        )

        self._jobs[job_id] = job
        logger.info("Scheduled SFT job: %s", job_id)

        # Start job in background
        asyncio.create_task(self._run_sft_job(job))

        return job

    async def schedule_dpo(
        self,
        model: str | TinkerModel = TinkerModel.LLAMA_3_3_70B,
        adapter_name: str | None = None,
        min_elo_difference: float | None = None,
        limit: int | None = None,
        beta: float = 0.1,
        **kwargs,
    ) -> TrainingJob:
        """
        Schedule a DPO training job.

        Args:
            model: Base model to fine-tune
            adapter_name: Name for the adapter
            min_elo_difference: Minimum ELO gap for preference pairs
            limit: Maximum training examples
            beta: DPO temperature parameter
            **kwargs: Additional training config

        Returns:
            TrainingJob with job details
        """
        if isinstance(model, TinkerModel):
            model = model.value

        job_id = self._generate_job_id()
        adapter_name = adapter_name or f"aragora-dpo-{job_id}"

        job = TrainingJob(
            job_id=job_id,
            job_type=JobType.DPO,
            model=model,
            config={
                "adapter_name": adapter_name,
                "min_elo_difference": min_elo_difference or self.config.dpo_min_elo_difference,
                "limit": limit or self.config.export_limit,
                "beta": beta,
                **kwargs,
            },
        )

        self._jobs[job_id] = job
        logger.info("Scheduled DPO job: %s", job_id)

        asyncio.create_task(self._run_dpo_job(job))

        return job

    async def schedule_combined(
        self,
        model: str | TinkerModel = TinkerModel.LLAMA_3_3_70B,
        adapter_name: str | None = None,
        **kwargs,
    ) -> TrainingJob:
        """
        Schedule a combined SFT + DPO training pipeline.

        First trains SFT on winning patterns, then refines with DPO
        on preference pairs.

        Args:
            model: Base model to fine-tune
            adapter_name: Name for the adapter
            **kwargs: Additional training config

        Returns:
            TrainingJob with job details
        """
        if isinstance(model, TinkerModel):
            model = model.value

        job_id = self._generate_job_id()
        adapter_name = adapter_name or f"aragora-combined-{job_id}"

        job = TrainingJob(
            job_id=job_id,
            job_type=JobType.COMBINED,
            model=model,
            config={
                "adapter_name": adapter_name,
                "sft_limit": kwargs.get("sft_limit", self.config.export_limit),
                "dpo_limit": kwargs.get("dpo_limit", self.config.export_limit // 2),
                **kwargs,
            },
        )

        self._jobs[job_id] = job
        logger.info("Scheduled combined job: %s", job_id)

        asyncio.create_task(self._run_combined_job(job))

        return job

    async def _run_sft_job(self, job: TrainingJob) -> None:
        """Run an SFT training job."""
        try:
            job.status = JobStatus.PREPARING
            job.started_at = datetime.now().isoformat()

            # Export training data
            exporter = SFTExporter()
            data = exporter.export(
                min_confidence=job.config.get("min_confidence", 0.7),
                limit=job.config.get("limit", 1000),
            )

            if not data:
                raise ValueError("No training data exported")

            # Save data to file
            data_path = self.config.data_dir / f"{job.job_id}_sft.jsonl"
            with open(data_path, "w") as f:
                for record in data:
                    f.write(json.dumps(record) + "\n")

            logger.info("Exported %d SFT records for job %s", len(data), job.job_id)

            # Submit to Tinker
            job.status = JobStatus.SUBMITTED
            result = await self.client.train_sft(
                training_data=data,
                model=job.model,
                adapter_name=job.config.get("adapter_name"),
            )

            job.tinker_job_id = result.job_id
            job.status = JobStatus.RUNNING if result.state == TrainingState.RUNNING else JobStatus.COMPLETED
            job.result = result

            if result.state == TrainingState.COMPLETED:
                job.model_id = result.model_id
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now().isoformat()
                logger.info("SFT job %s completed: model_id=%s", job.job_id, result.model_id)
            elif result.state == TrainingState.FAILED:
                job.status = JobStatus.FAILED
                job.error = result.error_message
                job.completed_at = datetime.now().isoformat()
                logger.error("SFT job %s failed: %s", job.job_id, result.error_message)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now().isoformat()
            logger.exception("SFT job %s failed with exception", job.job_id)

    async def _run_dpo_job(self, job: TrainingJob) -> None:
        """Run a DPO training job."""
        try:
            job.status = JobStatus.PREPARING
            job.started_at = datetime.now().isoformat()

            # Export preference data
            exporter = DPOExporter()
            data = exporter.export(
                min_elo_difference=job.config.get("min_elo_difference", 50.0),
                limit=job.config.get("limit", 500),
            )

            if not data:
                raise ValueError("No preference data exported")

            # Save data to file
            data_path = self.config.data_dir / f"{job.job_id}_dpo.jsonl"
            with open(data_path, "w") as f:
                for record in data:
                    f.write(json.dumps(record) + "\n")

            logger.info("Exported %d DPO records for job %s", len(data), job.job_id)

            # Submit to Tinker
            job.status = JobStatus.SUBMITTED
            result = await self.client.train_dpo(
                preference_data=data,
                model=job.model,
                adapter_name=job.config.get("adapter_name"),
                beta=job.config.get("beta", 0.1),
            )

            job.tinker_job_id = result.job_id
            job.result = result

            if result.state == TrainingState.COMPLETED:
                job.model_id = result.model_id
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now().isoformat()
                logger.info("DPO job %s completed: model_id=%s", job.job_id, result.model_id)
            elif result.state == TrainingState.FAILED:
                job.status = JobStatus.FAILED
                job.error = result.error_message
                job.completed_at = datetime.now().isoformat()
                logger.error("DPO job %s failed: %s", job.job_id, result.error_message)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now().isoformat()
            logger.exception("DPO job %s failed with exception", job.job_id)

    async def _run_combined_job(self, job: TrainingJob) -> None:
        """Run a combined SFT + DPO pipeline."""
        try:
            job.status = JobStatus.PREPARING
            job.started_at = datetime.now().isoformat()

            # Phase 1: SFT
            sft_exporter = SFTExporter()
            sft_data = sft_exporter.export(limit=job.config.get("sft_limit", 1000))

            if not sft_data:
                raise ValueError("No SFT training data exported")

            job.status = JobStatus.SUBMITTED
            sft_result = await self.client.train_sft(
                training_data=sft_data,
                model=job.model,
                adapter_name=f"{job.config.get('adapter_name')}-sft",
            )

            if sft_result.state != TrainingState.COMPLETED:
                raise ValueError(f"SFT phase failed: {sft_result.error_message}")

            logger.info("Combined job %s: SFT phase complete", job.job_id)

            # Phase 2: DPO (using SFT model as base)
            dpo_exporter = DPOExporter()
            dpo_data = dpo_exporter.export(limit=job.config.get("dpo_limit", 500))

            if dpo_data:
                dpo_result = await self.client.train_dpo(
                    preference_data=dpo_data,
                    model=sft_result.model_id,  # Use SFT output as base
                    adapter_name=job.config.get("adapter_name"),
                )

                job.result = dpo_result
                job.model_id = dpo_result.model_id

                if dpo_result.state == TrainingState.COMPLETED:
                    job.status = JobStatus.COMPLETED
                    logger.info("Combined job %s completed: model_id=%s", job.job_id, dpo_result.model_id)
                else:
                    job.status = JobStatus.FAILED
                    job.error = dpo_result.error_message
            else:
                # No DPO data, use SFT result
                job.result = sft_result
                job.model_id = sft_result.model_id
                job.status = JobStatus.COMPLETED
                logger.info("Combined job %s completed (SFT only): model_id=%s", job.job_id, sft_result.model_id)

            job.completed_at = datetime.now().isoformat()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now().isoformat()
            logger.exception("Combined job %s failed with exception", job.job_id)

    def get_job(self, job_id: str) -> TrainingJob | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
        limit: int = 50,
    ) -> list[TrainingJob]:
        """List scheduled jobs."""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]

        # Sort by creation time, newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ) -> TrainingJob:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait

        Returns:
            Completed TrainingJob

        Raises:
            TimeoutError: If job doesn't complete within timeout
            ValueError: If job not found
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        start_time = asyncio.get_event_loop().time()

        while job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            await asyncio.sleep(poll_interval)
            job = self._jobs[job_id]  # Refresh

        return job

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self.get_job(job_id)
        if not job:
            return False

        if job.status in (JobStatus.PENDING, JobStatus.PREPARING):
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            logger.info("Cancelled job: %s", job_id)
            return True

        return False

    def save_state(self, path: Path | str) -> None:
        """Save scheduler state to file."""
        path = Path(path)
        state = {
            "jobs": [job.to_dict() for job in self._jobs.values()],
            "job_counter": self._job_counter,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path | str) -> None:
        """Load scheduler state from file."""
        path = Path(path)
        if not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        self._job_counter = state.get("job_counter", 0)

        for job_dict in state.get("jobs", []):
            job = TrainingJob(
                job_id=job_dict["job_id"],
                job_type=JobType(job_dict["job_type"]),
                model=job_dict["model"],
                status=JobStatus(job_dict["status"]),
                created_at=job_dict.get("created_at", ""),
                started_at=job_dict.get("started_at"),
                completed_at=job_dict.get("completed_at"),
                tinker_job_id=job_dict.get("tinker_job_id"),
                model_id=job_dict.get("model_id"),
                config=job_dict.get("config", {}),
                error=job_dict.get("error"),
            )
            self._jobs[job.job_id] = job

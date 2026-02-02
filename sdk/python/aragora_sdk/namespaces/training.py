"""
Training Namespace API

Provides model training data export and job management:
- SFT (Supervised Fine-Tuning) data export
- DPO (Direct Preference Optimization) data export
- Gauntlet adversarial data export
- Training job management and metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

# Type aliases
ExportFormat = Literal["json", "jsonl"]
ExportType = Literal["sft", "dpo", "gauntlet"]
JobStatus = Literal["pending", "training", "completed", "failed", "cancelled"]
GauntletPersona = Literal["gdpr", "hipaa", "ai_act", "all"]


class TrainingAPI:
    """
    Synchronous Training API.

    Provides methods for exporting training data and managing training jobs:
    - Export SFT, DPO, and Gauntlet data for model training
    - Manage training jobs lifecycle
    - Retrieve training metrics and artifacts

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> sft_data = client.training.export_sft(min_confidence=0.8, limit=5000)
        >>> dpo_data = client.training.export_dpo(min_confidence_diff=0.2)
        >>> jobs = client.training.list_jobs(status="completed")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Data Export
    # ===========================================================================

    def export_sft(
        self,
        min_confidence: float = 0.7,
        min_success_rate: float = 0.6,
        limit: int = 1000,
        offset: int = 0,
        include_critiques: bool = True,
        include_patterns: bool = True,
        include_debates: bool = True,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export SFT (Supervised Fine-Tuning) training data.

        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0)
            min_success_rate: Minimum success rate threshold (0.0-1.0)
            limit: Maximum records to export
            offset: Offset for pagination
            include_critiques: Include critique data
            include_patterns: Include pattern data
            include_debates: Include debate data
            format: Export format ('json' or 'jsonl')

        Returns:
            Dict with export_type, total_records, parameters, exported_at, format, and records/data
        """
        params: dict[str, Any] = {
            "min_confidence": min_confidence,
            "min_success_rate": min_success_rate,
            "limit": limit,
            "offset": offset,
            "include_critiques": include_critiques,
            "include_patterns": include_patterns,
            "include_debates": include_debates,
            "format": format,
        }
        return self._client.request("POST", "/api/v1/training/export/sft", params=params)

    def export_dpo(
        self,
        min_confidence_diff: float = 0.1,
        limit: int = 500,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export DPO (Direct Preference Optimization) training data.

        Args:
            min_confidence_diff: Minimum confidence difference between chosen/rejected
            limit: Maximum records to export
            format: Export format

        Returns:
            Dict with export_type, total_records, parameters, exported_at, format, and records/data
        """
        params: dict[str, Any] = {
            "min_confidence_diff": min_confidence_diff,
            "limit": limit,
            "format": format,
        }
        return self._client.request("POST", "/api/v1/training/export/dpo", params=params)

    def export_gauntlet(
        self,
        persona: GauntletPersona = "all",
        min_severity: int | None = None,
        limit: int | None = None,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export Gauntlet adversarial training data.

        Args:
            persona: Persona type to export ('gdpr', 'hipaa', 'ai_act', 'all')
            min_severity: Minimum severity threshold
            limit: Maximum records to export
            format: Export format

        Returns:
            Dict with export_type, total_records, parameters, exported_at, format, and records/data
        """
        params: dict[str, Any] = {
            "persona": persona,
            "format": format,
        }
        if min_severity is not None:
            params["min_severity"] = min_severity
        if limit is not None:
            params["limit"] = limit
        return self._client.request("POST", "/api/v1/training/export/gauntlet", params=params)

    # ===========================================================================
    # Statistics and Formats
    # ===========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get training data statistics.

        Returns:
            Dict with stats containing total_debates, total_patterns, total_critiques,
            sft_eligible, dpo_eligible, gauntlet_eligible, and last_updated
        """
        return self._client.request("GET", "/api/v1/training/stats")

    def get_formats(self) -> dict[str, Any]:
        """
        Get supported training formats and schemas.

        Returns:
            Dict with formats containing supported_formats, sft_schema, dpo_schema,
            and gauntlet_schema
        """
        return self._client.request("GET", "/api/v1/training/formats")

    # ===========================================================================
    # Job Management
    # ===========================================================================

    def list_jobs(
        self,
        limit: int | None = None,
        offset: int | None = None,
        status: JobStatus | None = None,
        vertical: str | None = None,
    ) -> dict[str, Any]:
        """
        List training jobs.

        Args:
            limit: Maximum jobs to return
            offset: Pagination offset
            status: Filter by job status
            vertical: Filter by vertical

        Returns:
            Dict with jobs array, total, limit, and offset
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if status is not None:
            params["status"] = status
        if vertical is not None:
            params["vertical"] = vertical
        return self._client.request("GET", "/api/v1/training/jobs", params=params or None)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """
        Get training job details.

        Args:
            job_id: Job ID

        Returns:
            Dict with job details including id, status, config, hyperparameters, checkpoints
        """
        return self._client.request("GET", f"/api/v1/training/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """
        Cancel a training job.

        Args:
            job_id: Job ID

        Returns:
            Dict with success and message
        """
        return self._client.request("DELETE", f"/api/v1/training/jobs/{job_id}")

    def export_job_data(self, job_id: str) -> dict[str, Any]:
        """
        Export training job data.

        Args:
            job_id: Job ID

        Returns:
            Dict with success, examples_exported, and export_path
        """
        return self._client.request("POST", f"/api/v1/training/jobs/{job_id}/export")

    def start_job(self, job_id: str) -> dict[str, Any]:
        """
        Start a training job.

        Args:
            job_id: Job ID

        Returns:
            Dict with success, training_job_id, and message
        """
        return self._client.request("POST", f"/api/v1/training/jobs/{job_id}/start")

    def complete_job(
        self,
        job_id: str,
        final_loss: float | None = None,
        elo_rating: float | None = None,
        win_rate: float | None = None,
        vertical_accuracy: float | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Mark a training job as complete (typically called by webhook).

        Args:
            job_id: Job ID
            final_loss: Final training loss
            elo_rating: Resulting ELO rating
            win_rate: Win rate against baseline
            vertical_accuracy: Vertical-specific accuracy
            artifacts: Artifact paths

        Returns:
            Dict with success and message
        """
        data: dict[str, Any] = {}
        if final_loss is not None:
            data["final_loss"] = final_loss
        if elo_rating is not None:
            data["elo_rating"] = elo_rating
        if win_rate is not None:
            data["win_rate"] = win_rate
        if vertical_accuracy is not None:
            data["vertical_accuracy"] = vertical_accuracy
        if artifacts is not None:
            data["artifacts"] = artifacts
        return self._client.request(
            "POST", f"/api/v1/training/jobs/{job_id}/complete", json=data or None
        )

    def get_job_metrics(self, job_id: str) -> dict[str, Any]:
        """
        Get training job metrics.

        Args:
            job_id: Job ID

        Returns:
            Dict with job_id, status, training_data_examples, metrics_history, etc.
        """
        return self._client.request("GET", f"/api/v1/training/jobs/{job_id}/metrics")

    def get_job_artifacts(self, job_id: str) -> dict[str, Any]:
        """
        Get training job artifacts.

        Args:
            job_id: Job ID

        Returns:
            Dict with job_id, adapter_path, config_path, metrics_path, checkpoints, total_size_bytes
        """
        return self._client.request("GET", f"/api/v1/training/jobs/{job_id}/artifacts")


class AsyncTrainingAPI:
    """
    Asynchronous Training API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     sft_data = await client.training.export_sft(min_confidence=0.8)
        ...     jobs = await client.training.list_jobs(status="completed")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Data Export
    # ===========================================================================

    async def export_sft(
        self,
        min_confidence: float = 0.7,
        min_success_rate: float = 0.6,
        limit: int = 1000,
        offset: int = 0,
        include_critiques: bool = True,
        include_patterns: bool = True,
        include_debates: bool = True,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """Export SFT (Supervised Fine-Tuning) training data."""
        params: dict[str, Any] = {
            "min_confidence": min_confidence,
            "min_success_rate": min_success_rate,
            "limit": limit,
            "offset": offset,
            "include_critiques": include_critiques,
            "include_patterns": include_patterns,
            "include_debates": include_debates,
            "format": format,
        }
        return await self._client.request("POST", "/api/v1/training/export/sft", params=params)

    async def export_dpo(
        self,
        min_confidence_diff: float = 0.1,
        limit: int = 500,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """Export DPO (Direct Preference Optimization) training data."""
        params: dict[str, Any] = {
            "min_confidence_diff": min_confidence_diff,
            "limit": limit,
            "format": format,
        }
        return await self._client.request("POST", "/api/v1/training/export/dpo", params=params)

    async def export_gauntlet(
        self,
        persona: GauntletPersona = "all",
        min_severity: int | None = None,
        limit: int | None = None,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """Export Gauntlet adversarial training data."""
        params: dict[str, Any] = {
            "persona": persona,
            "format": format,
        }
        if min_severity is not None:
            params["min_severity"] = min_severity
        if limit is not None:
            params["limit"] = limit
        return await self._client.request("POST", "/api/v1/training/export/gauntlet", params=params)

    # ===========================================================================
    # Statistics and Formats
    # ===========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get training data statistics."""
        return await self._client.request("GET", "/api/v1/training/stats")

    async def get_formats(self) -> dict[str, Any]:
        """Get supported training formats and schemas."""
        return await self._client.request("GET", "/api/v1/training/formats")

    # ===========================================================================
    # Job Management
    # ===========================================================================

    async def list_jobs(
        self,
        limit: int | None = None,
        offset: int | None = None,
        status: JobStatus | None = None,
        vertical: str | None = None,
    ) -> dict[str, Any]:
        """List training jobs."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if status is not None:
            params["status"] = status
        if vertical is not None:
            params["vertical"] = vertical
        return await self._client.request("GET", "/api/v1/training/jobs", params=params or None)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        """Get training job details."""
        return await self._client.request("GET", f"/api/v1/training/jobs/{job_id}")

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a training job."""
        return await self._client.request("DELETE", f"/api/v1/training/jobs/{job_id}")

    async def export_job_data(self, job_id: str) -> dict[str, Any]:
        """Export training job data."""
        return await self._client.request("POST", f"/api/v1/training/jobs/{job_id}/export")

    async def start_job(self, job_id: str) -> dict[str, Any]:
        """Start a training job."""
        return await self._client.request("POST", f"/api/v1/training/jobs/{job_id}/start")

    async def complete_job(
        self,
        job_id: str,
        final_loss: float | None = None,
        elo_rating: float | None = None,
        win_rate: float | None = None,
        vertical_accuracy: float | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Mark a training job as complete."""
        data: dict[str, Any] = {}
        if final_loss is not None:
            data["final_loss"] = final_loss
        if elo_rating is not None:
            data["elo_rating"] = elo_rating
        if win_rate is not None:
            data["win_rate"] = win_rate
        if vertical_accuracy is not None:
            data["vertical_accuracy"] = vertical_accuracy
        if artifacts is not None:
            data["artifacts"] = artifacts
        return await self._client.request(
            "POST", f"/api/v1/training/jobs/{job_id}/complete", json=data or None
        )

    async def get_job_metrics(self, job_id: str) -> dict[str, Any]:
        """Get training job metrics."""
        return await self._client.request("GET", f"/api/v1/training/jobs/{job_id}/metrics")

    async def get_job_artifacts(self, job_id: str) -> dict[str, Any]:
        """Get training job artifacts."""
        return await self._client.request("GET", f"/api/v1/training/jobs/{job_id}/artifacts")

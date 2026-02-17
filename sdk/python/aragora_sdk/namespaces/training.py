"""
Training Namespace API

Provides model training data export and job management:
- SFT (Supervised Fine-Tuning) data export
- DPO (Direct Preference Optimization) data export
- Gauntlet adversarial data export
- Training job management and metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


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

    def export_sft(
        self,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Export SFT (Supervised Fine-Tuning) training data."""
        params: dict[str, Any] = {}
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/training/export/sft", params=params or None)

    def export_dpo(
        self,
        min_confidence_diff: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Export DPO (Direct Preference Optimization) training data."""
        params: dict[str, Any] = {}
        if min_confidence_diff is not None:
            params["min_confidence_diff"] = min_confidence_diff
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/training/export/dpo", params=params or None)

    def export_gauntlet(
        self,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Export Gauntlet adversarial training data."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/training/export/gauntlet", params=params or None)

    def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        return self._client.request("GET", "/api/v1/training/stats")

    def get_formats(self) -> dict[str, Any]:
        """Get available training data formats."""
        return self._client.request("GET", "/api/v1/training/formats")

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List training jobs."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/training/jobs", params=params)


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

    async def export_sft(
        self,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Export SFT (Supervised Fine-Tuning) training data."""
        params: dict[str, Any] = {}
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        if limit is not None:
            params["limit"] = limit
        return await self._client.request("GET", "/api/v1/training/export/sft", params=params or None)

    async def export_dpo(
        self,
        min_confidence_diff: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Export DPO (Direct Preference Optimization) training data."""
        params: dict[str, Any] = {}
        if min_confidence_diff is not None:
            params["min_confidence_diff"] = min_confidence_diff
        if limit is not None:
            params["limit"] = limit
        return await self._client.request("GET", "/api/v1/training/export/dpo", params=params or None)

    async def export_gauntlet(
        self,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Export Gauntlet adversarial training data."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        return await self._client.request("GET", "/api/v1/training/export/gauntlet", params=params or None)

    async def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        return await self._client.request("GET", "/api/v1/training/stats")

    async def get_formats(self) -> dict[str, Any]:
        """Get available training data formats."""
        return await self._client.request("GET", "/api/v1/training/formats")

    async def list_jobs(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List training jobs."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/training/jobs", params=params)

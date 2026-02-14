"""
Training Namespace API

Provides model training data export and job management:
- SFT (Supervised Fine-Tuning) data export
- DPO (Direct Preference Optimization) data export
- Gauntlet adversarial data export
- Training job management and metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

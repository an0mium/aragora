"""
Batch Namespace API.

Provides bulk operations for debates and other resources:
- Batch debate submission
- Batch status tracking
- Batch result retrieval
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

BatchStatus = Literal["pending", "processing", "completed", "failed", "cancelled"]
BatchOperationType = Literal["debate", "review", "analysis", "export"]


class BatchAPI:
    """
    Synchronous Batch API.

    Provides methods for bulk operations:
    - Submit batch debates
    - Track batch progress
    - Retrieve batch results

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> batch = client.batch.submit_debates(
        ...     debates=[
        ...         {"task": "Review microservices architecture"},
        ...         {"task": "Evaluate caching strategy"}
        ...     ]
        ... )
        >>> status = client.batch.get_status(batch["batch_id"])
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def submit_debates(
        self,
        debates: list[dict[str, Any]],
        priority: str | None = None,
        callback_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a batch of debates for processing.

        Args:
            debates: List of debate configurations.
            priority: Batch priority ('low', 'normal', 'high').
            callback_url: Webhook URL for completion notification.
            metadata: Custom metadata for the batch.

        Returns:
            Batch submission response with batch_id.
        """
        data: dict[str, Any] = {"debates": debates}
        if priority:
            data["priority"] = priority
        if callback_url:
            data["callback_url"] = callback_url
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/v1/batch/debates", json=data)

    def get_status(self, batch_id: str) -> dict[str, Any]:
        """
        Get the status of a batch operation.

        Args:
            batch_id: Batch identifier.

        Returns:
            Batch status with progress information.
        """
        return self._client.request("GET", f"/api/v1/batch/{batch_id}/status")

    def get_results(
        self,
        batch_id: str,
        include_failed: bool = False,
    ) -> dict[str, Any]:
        """
        Get results of a completed batch.

        Args:
            batch_id: Batch identifier.
            include_failed: Include failed items in results.

        Returns:
            Batch results with individual item outcomes.
        """
        params: dict[str, Any] = {}
        if include_failed:
            params["include_failed"] = include_failed

        return self._client.request(
            "GET", f"/api/v1/batch/{batch_id}/results", params=params or None
        )

    def list(
        self,
        status: BatchStatus | None = None,
        operation_type: BatchOperationType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List batch operations.

        Args:
            status: Filter by status.
            operation_type: Filter by operation type.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Paginated list of batches.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if operation_type:
            params["operation_type"] = operation_type

        return self._client.request("GET", "/api/v1/batch", params=params)

    def cancel(self, batch_id: str) -> dict[str, Any]:
        """
        Cancel a batch operation.

        Args:
            batch_id: Batch identifier.

        Returns:
            Cancellation confirmation.
        """
        return self._client.request("POST", f"/api/v1/batch/{batch_id}/cancel")

    def delete(self, batch_id: str) -> dict[str, Any]:
        """
        Delete a batch and its results.

        Args:
            batch_id: Batch identifier.

        Returns:
            Deletion confirmation.
        """
        return self._client.request("DELETE", f"/api/v1/batch/{batch_id}")


class AsyncBatchAPI:
    """Asynchronous Batch API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def submit_debates(
        self,
        debates: list[dict[str, Any]],
        priority: str | None = None,
        callback_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of debates for processing."""
        data: dict[str, Any] = {"debates": debates}
        if priority:
            data["priority"] = priority
        if callback_url:
            data["callback_url"] = callback_url
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("POST", "/api/v1/batch/debates", json=data)

    async def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get the status of a batch operation."""
        return await self._client.request("GET", f"/api/v1/batch/{batch_id}/status")

    async def get_results(
        self,
        batch_id: str,
        include_failed: bool = False,
    ) -> dict[str, Any]:
        """Get results of a completed batch."""
        params: dict[str, Any] = {}
        if include_failed:
            params["include_failed"] = include_failed

        return await self._client.request(
            "GET", f"/api/v1/batch/{batch_id}/results", params=params or None
        )

    async def list(
        self,
        status: BatchStatus | None = None,
        operation_type: BatchOperationType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List batch operations."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if operation_type:
            params["operation_type"] = operation_type

        return await self._client.request("GET", "/api/v1/batch", params=params)

    async def cancel(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch operation."""
        return await self._client.request("POST", f"/api/v1/batch/{batch_id}/cancel")

    async def delete(self, batch_id: str) -> dict[str, Any]:
        """Delete a batch and its results."""
        return await self._client.request("DELETE", f"/api/v1/batch/{batch_id}")

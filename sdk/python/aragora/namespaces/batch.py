"""
Batch Namespace API.

Provides bulk operations for debates:
- Batch debate submission
- Batch status tracking
- Batch listing and queue visibility
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
    - List batches

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
        items = []
        for item in debates:
            if priority and "priority" not in item:
                item = {**item, "priority": priority}
            if metadata and "metadata" not in item:
                item = {**item, "metadata": metadata}
            items.append(item)

        data: dict[str, Any] = {"items": items}
        if priority:
            data["priority"] = priority
        if callback_url:
            data["webhook_url"] = callback_url
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/v1/debates/batch", json=data)

    def get_status(self, batch_id: str) -> dict[str, Any]:
        """
        Get the status of a batch operation.

        Args:
            batch_id: Batch identifier.

        Returns:
            Batch status with progress information.
        """
        return self._client.request("GET", f"/api/v1/debates/batch/{batch_id}/status")

    def get_results(
        self,
        batch_id: str,
        include_failed: bool = False,
    ) -> dict[str, Any]:
        """
        Get results of a completed batch.

        Note: The batch status endpoint includes per-item results; this method
        is an alias for `get_status` for API compatibility.
        """
        _ = include_failed
        return self._client.request("GET", f"/api/v1/debates/batch/{batch_id}/status")

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

        return self._client.request("GET", "/api/v1/debates/batch", params=params)

    def cancel(self, batch_id: str) -> dict[str, Any]:
        """
        Cancel a batch operation.

        Args:
            batch_id: Batch identifier.

        Returns:
            Cancellation confirmation.
        """
        raise NotImplementedError("Batch cancel is not exposed via the public API")

    def delete(self, batch_id: str) -> dict[str, Any]:
        """
        Delete a batch and its results.

        Args:
            batch_id: Batch identifier.

        Returns:
            Deletion confirmation.
        """
        raise NotImplementedError("Batch delete is not exposed via the public API")


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

        return await self._client.request("POST", "/api/v1/debates/batch", json=data)

    async def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get the status of a batch operation."""
        return await self._client.request("GET", f"/api/v1/debates/batch/{batch_id}/status")

    async def get_results(
        self,
        batch_id: str,
        include_failed: bool = False,
    ) -> dict[str, Any]:
        """Get results of a completed batch."""
        _ = include_failed
        return await self._client.request("GET", f"/api/v1/debates/batch/{batch_id}/status")

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

        return await self._client.request("GET", "/api/v1/debates/batch", params=params)

    async def cancel(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch operation."""
        raise NotImplementedError("Batch cancel is not exposed via the public API")

    async def delete(self, batch_id: str) -> dict[str, Any]:
        """Delete a batch and its results."""
        raise NotImplementedError("Batch delete is not exposed via the public API")

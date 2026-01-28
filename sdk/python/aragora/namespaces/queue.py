"""
Queue Namespace API.

Provides access to queue management endpoints for background jobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class QueueAPI:
    """Synchronous queue API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list_jobs(self, **params: Any) -> dict[str, Any]:
        return self._client.request("GET", "/api/queue/jobs", params=params)

    def enqueue(self, job_type: str, payload: dict[str, Any], **params: Any) -> dict[str, Any]:
        data = {"type": job_type, "payload": payload, **params}
        return self._client.request("POST", "/api/queue/jobs", json=data)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._client.request("GET", f"/api/queue/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._client.request("DELETE", f"/api/queue/jobs/{job_id}")

    def retry_job(self, job_id: str) -> dict[str, Any]:
        return self._client.request("POST", f"/api/queue/jobs/{job_id}/retry")

    def get_stats(self) -> dict[str, Any]:
        return self._client.request("GET", "/api/queue/stats")

    def list_workers(self) -> dict[str, Any]:
        return self._client.request("GET", "/api/queue/workers")

    def list_dlq(self, limit: int = 50) -> dict[str, Any]:
        return self._client.request("GET", "/api/queue/dlq", params={"limit": limit})

    def requeue_dlq(self, job_ids: list[str] | None = None) -> dict[str, Any]:
        payload = {"job_ids": job_ids or []}
        return self._client.request("POST", "/api/queue/dlq/requeue", json=payload)

    def cleanup(self, older_than_days: int = 7, status: list[str] | None = None) -> dict[str, Any]:
        payload = {"older_than_days": older_than_days}
        if status is not None:
            payload["status"] = status
        return self._client.request("POST", "/api/queue/cleanup", json=payload)

    def list_stale(self, threshold_minutes: int = 30) -> dict[str, Any]:
        return self._client.request(
            "GET",
            "/api/queue/stale",
            params={"threshold_minutes": threshold_minutes},
        )


class AsyncQueueAPI:
    """Asynchronous queue API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list_jobs(self, **params: Any) -> dict[str, Any]:
        return await self._client.request("GET", "/api/queue/jobs", params=params)

    async def enqueue(
        self, job_type: str, payload: dict[str, Any], **params: Any
    ) -> dict[str, Any]:
        data = {"type": job_type, "payload": payload, **params}
        return await self._client.request("POST", "/api/queue/jobs", json=data)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._client.request("GET", f"/api/queue/jobs/{job_id}")

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self._client.request("DELETE", f"/api/queue/jobs/{job_id}")

    async def retry_job(self, job_id: str) -> dict[str, Any]:
        return await self._client.request("POST", f"/api/queue/jobs/{job_id}/retry")

    async def get_stats(self) -> dict[str, Any]:
        return await self._client.request("GET", "/api/queue/stats")

    async def list_workers(self) -> dict[str, Any]:
        return await self._client.request("GET", "/api/queue/workers")

    async def list_dlq(self, limit: int = 50) -> dict[str, Any]:
        return await self._client.request("GET", "/api/queue/dlq", params={"limit": limit})

    async def requeue_dlq(self, job_ids: list[str] | None = None) -> dict[str, Any]:
        payload = {"job_ids": job_ids or []}
        return await self._client.request("POST", "/api/queue/dlq/requeue", json=payload)

    async def cleanup(
        self, older_than_days: int = 7, status: list[str] | None = None
    ) -> dict[str, Any]:
        payload = {"older_than_days": older_than_days}
        if status is not None:
            payload["status"] = status
        return await self._client.request("POST", "/api/queue/cleanup", json=payload)

    async def list_stale(self, threshold_minutes: int = 30) -> dict[str, Any]:
        return await self._client.request(
            "GET",
            "/api/queue/stale",
            params={"threshold_minutes": threshold_minutes},
        )

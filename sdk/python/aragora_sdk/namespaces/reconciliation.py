"""
Reconciliation Namespace API

Provides methods for data reconciliation:
- Compare data sources
- Identify discrepancies
- Generate reconciliation reports
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ReconciliationAPI:
    """Synchronous Reconciliation API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def create(
        self,
        source_a: dict[str, Any],
        source_b: dict[str, Any],
        rules: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a reconciliation job."""
        data: dict[str, Any] = {"source_a": source_a, "source_b": source_b}
        if rules:
            data["rules"] = rules
        return self._client.request("POST", "/api/v1/reconciliation", json=data)

    def list(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List reconciliation jobs."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/reconciliation", params=params)

    def get(self, job_id: str) -> dict[str, Any]:
        """Get reconciliation job by ID."""
        return self._client.request("GET", f"/api/v1/reconciliation/{job_id}")

    def get_discrepancies(self, job_id: str, severity: str | None = None) -> dict[str, Any]:
        """Get discrepancies from a job."""
        params: dict[str, Any] = {}
        if severity:
            params["severity"] = severity
        return self._client.request(
            "GET", f"/api/v1/reconciliation/{job_id}/discrepancies", params=params
        )

    def resolve_discrepancy(
        self, job_id: str, discrepancy_id: str, resolution: str
    ) -> dict[str, Any]:
        """Resolve a discrepancy."""
        return self._client.request(
            "POST",
            f"/api/v1/reconciliation/{job_id}/discrepancies/{discrepancy_id}/resolve",
            json={
                "resolution": resolution,
            },
        )

    def get_report(self, job_id: str, format: str = "json") -> dict[str, Any]:
        """Get reconciliation report."""
        return self._client.request(
            "GET", f"/api/v1/reconciliation/{job_id}/report", params={"format": format}
        )

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a reconciliation job."""
        return self._client.request("POST", f"/api/v1/reconciliation/{job_id}/cancel")


class AsyncReconciliationAPI:
    """Asynchronous Reconciliation API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create(
        self,
        source_a: dict[str, Any],
        source_b: dict[str, Any],
        rules: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a reconciliation job."""
        data: dict[str, Any] = {"source_a": source_a, "source_b": source_b}
        if rules:
            data["rules"] = rules
        return await self._client.request("POST", "/api/v1/reconciliation", json=data)

    async def list(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List reconciliation jobs."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/reconciliation", params=params)

    async def get(self, job_id: str) -> dict[str, Any]:
        """Get reconciliation job by ID."""
        return await self._client.request("GET", f"/api/v1/reconciliation/{job_id}")

    async def get_discrepancies(self, job_id: str, severity: str | None = None) -> dict[str, Any]:
        """Get discrepancies from a job."""
        params: dict[str, Any] = {}
        if severity:
            params["severity"] = severity
        return await self._client.request(
            "GET", f"/api/v1/reconciliation/{job_id}/discrepancies", params=params
        )

    async def resolve_discrepancy(
        self, job_id: str, discrepancy_id: str, resolution: str
    ) -> dict[str, Any]:
        """Resolve a discrepancy."""
        return await self._client.request(
            "POST",
            f"/api/v1/reconciliation/{job_id}/discrepancies/{discrepancy_id}/resolve",
            json={
                "resolution": resolution,
            },
        )

    async def get_report(self, job_id: str, format: str = "json") -> dict[str, Any]:
        """Get reconciliation report."""
        return await self._client.request(
            "GET", f"/api/v1/reconciliation/{job_id}/report", params={"format": format}
        )

    async def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a reconciliation job."""
        return await self._client.request("POST", f"/api/v1/reconciliation/{job_id}/cancel")

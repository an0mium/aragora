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

    def get(self, job_id: str) -> dict[str, Any]:
        """Get reconciliation job by ID."""
        return self._client.request("GET", f"/api/v1/reconciliation/{job_id}")

    def get_report(self, job_id: str, format: str = "json") -> dict[str, Any]:
        """Get reconciliation report."""
        return self._client.request(
            "GET", f"/api/v1/reconciliation/{job_id}/report", params={"format": format}
        )

class AsyncReconciliationAPI:
    """Asynchronous Reconciliation API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get(self, job_id: str) -> dict[str, Any]:
        """Get reconciliation job by ID."""
        return await self._client.request("GET", f"/api/v1/reconciliation/{job_id}")

    async def get_report(self, job_id: str, format: str = "json") -> dict[str, Any]:
        """Get reconciliation report."""
        return await self._client.request(
            "GET", f"/api/v1/reconciliation/{job_id}/report", params={"format": format}
        )


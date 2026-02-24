"""
Reconciliation Namespace API

Provides methods for data reconciliation:
- Run reconciliation between bank and book transactions
- View and manage discrepancies
- Resolve discrepancies with AI suggestions
- Generate reconciliation reports
- Approve reconciliation results
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ReconciliationAPI:
    """
    Synchronous Reconciliation API.

    Provides methods for bank reconciliation functionality including
    running reconciliations, managing discrepancies, and generating reports.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.reconciliation.run(
        ...     bank_account_id="acct-001",
        ...     start_date="2026-01-01",
        ...     end_date="2026-01-31",
        ... )
        >>> discrepancies = client.reconciliation.list_discrepancies()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Core Operations
    # =========================================================================

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Run a new reconciliation between bank and book transactions.

        Args:
            **kwargs: Reconciliation parameters including:
                - bank_account_id: Bank account identifier
                - start_date: Reconciliation start date (YYYY-MM-DD)
                - end_date: Reconciliation end date (YYYY-MM-DD)
                - match_threshold: Matching threshold (0.0-1.0)

        Returns:
            Dict with reconciliation results including job ID and summary.
        """
        return self._client.request("POST", "/api/v1/reconciliation/run", json=kwargs)

    def list_all(self, **kwargs: Any) -> dict[str, Any]:
        """
        List past reconciliation jobs.

        Returns:
            Dict with list of past reconciliations and their statuses.
        """
        return self._client.request("GET", "/api/v1/reconciliation/list", params=kwargs or None)

    def get(self, job_id: str) -> dict[str, Any]:
        """
        Get reconciliation job details by ID.

        Args:
            job_id: Reconciliation job identifier.

        Returns:
            Dict with reconciliation details.
        """
        return self._client.request("GET", f"/api/v1/reconciliation/{job_id}")

    def get_report(self, job_id: str, format: str = "json") -> dict[str, Any]:
        """
        Get reconciliation report.

        Args:
            job_id: Reconciliation job identifier.
            format: Report format (json, pdf, csv).

        Returns:
            Dict with reconciliation report data.
        """
        return self._client.request(
            "GET", f"/api/v1/reconciliation/{job_id}/report", params={"format": format}
        )

    def get_status(self) -> dict[str, Any]:
        """
        Get reconciliation status overview.

        Returns:
            Dict with reconciliation status information.
        """
        return self._client.request("GET", "/api/v1/reconciliation/status")

    def get_demo(self) -> dict[str, Any]:
        """
        Get demo reconciliation data for testing.

        Returns:
            Dict with sample reconciliation data.
        """
        return self._client.request("GET", "/api/v1/reconciliation/demo")

    # =========================================================================
    # Discrepancy Management
    # =========================================================================

    def list_discrepancies(self) -> dict[str, Any]:
        """
        List all pending discrepancies across reconciliations.

        Returns:
            Dict with pending discrepancies and their details.
        """
        return self._client.request("GET", "/api/v1/reconciliation/discrepancies")

    def resolve(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        """
        Resolve a discrepancy within a reconciliation.

        Args:
            job_id: Reconciliation job identifier.
            **kwargs: Resolution details including:
                - discrepancy_id: ID of the discrepancy to resolve
                - resolution: Resolution type (match, write_off, adjust)
                - notes: Resolution notes

        Returns:
            Dict with resolution confirmation.
        """
        return self._client.request("POST", f"/api/v1/reconciliation/{job_id}/resolve", json=kwargs)

    def bulk_resolve(self, **kwargs: Any) -> dict[str, Any]:
        """
        Bulk resolve multiple discrepancies.

        Args:
            **kwargs: Bulk resolution parameters including:
                - discrepancy_ids: List of discrepancy IDs
                - resolution: Resolution type to apply to all

        Returns:
            Dict with bulk resolution results.
        """
        return self._client.request(
            "POST", "/api/v1/reconciliation/discrepancies/bulk-resolve", json=kwargs
        )

    # =========================================================================
    # Approval
    # =========================================================================

    def approve(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        """
        Approve a completed reconciliation.

        Args:
            job_id: Reconciliation job identifier.
            **kwargs: Approval details including optional notes.

        Returns:
            Dict with approval confirmation.
        """
        return self._client.request("POST", f"/api/v1/reconciliation/{job_id}/approve", json=kwargs)


class AsyncReconciliationAPI:
    """
    Asynchronous Reconciliation API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.reconciliation.run(bank_account_id="acct-001")
        ...     discrepancies = await client.reconciliation.list_discrepancies()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run a new reconciliation between bank and book transactions."""
        return await self._client.request("POST", "/api/v1/reconciliation/run", json=kwargs)

    async def list_all(self, **kwargs: Any) -> dict[str, Any]:
        """List past reconciliation jobs."""
        return await self._client.request(
            "GET", "/api/v1/reconciliation/list", params=kwargs or None
        )

    async def get(self, job_id: str) -> dict[str, Any]:
        """Get reconciliation job details by ID."""
        return await self._client.request("GET", f"/api/v1/reconciliation/{job_id}")

    async def get_report(self, job_id: str, format: str = "json") -> dict[str, Any]:
        """Get reconciliation report."""
        return await self._client.request(
            "GET", f"/api/v1/reconciliation/{job_id}/report", params={"format": format}
        )

    async def get_status(self) -> dict[str, Any]:
        """Get reconciliation status overview."""
        return await self._client.request("GET", "/api/v1/reconciliation/status")

    async def get_demo(self) -> dict[str, Any]:
        """Get demo reconciliation data for testing."""
        return await self._client.request("GET", "/api/v1/reconciliation/demo")

    # =========================================================================
    # Discrepancy Management
    # =========================================================================

    async def list_discrepancies(self) -> dict[str, Any]:
        """List all pending discrepancies across reconciliations."""
        return await self._client.request("GET", "/api/v1/reconciliation/discrepancies")

    async def resolve(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        """Resolve a discrepancy within a reconciliation."""
        return await self._client.request(
            "POST", f"/api/v1/reconciliation/{job_id}/resolve", json=kwargs
        )

    async def bulk_resolve(self, **kwargs: Any) -> dict[str, Any]:
        """Bulk resolve multiple discrepancies."""
        return await self._client.request(
            "POST", "/api/v1/reconciliation/discrepancies/bulk-resolve", json=kwargs
        )

    # =========================================================================
    # Approval
    # =========================================================================

    async def approve(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        """Approve a completed reconciliation."""
        return await self._client.request(
            "POST", f"/api/v1/reconciliation/{job_id}/approve", json=kwargs
        )

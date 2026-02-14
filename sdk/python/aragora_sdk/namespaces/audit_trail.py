"""
Audit Trail Namespace API

Provides methods for audit trail access and verification:
- List and retrieve audit trails
- Export in multiple formats (json, csv, md)
- Verify integrity checksums
- Access v1 decision receipts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

AuditExportFormat = Literal["json", "csv", "md"]

class AuditTrailAPI:
    """
    Synchronous Audit Trail API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> trails = client.audit_trail.list()
        >>> for trail in trails["trails"]:
        ...     print(trail["trail_id"], trail["verdict"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # -- Audit trails ---------------------------------------------------------

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        List audit trails.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of audit trails.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset, **kwargs}
        return self._client.request("GET", "/api/v1/audit-trails", params=params)

    def get(self, trail_id: str) -> dict[str, Any]:
        """
        Get an audit trail by ID.

        Args:
            trail_id: Trail identifier.

        Returns:
            Audit trail details.
        """
        return self._client.request("GET", f"/api/v1/audit-trails/{trail_id}")

    def export(
        self,
        trail_id: str,
        format: AuditExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export an audit trail in the specified format.

        Args:
            trail_id: Trail identifier.
            format: Export format (json, csv, md).

        Returns:
            Exported trail data.
        """
        return self._client.request(
            "GET",
            f"/api/v1/audit-trails/{trail_id}/export",
            params={"format": format},
        )

    def verify(self, trail_id: str) -> dict[str, Any]:
        """
        Verify an audit trail's integrity checksum.

        Args:
            trail_id: Trail identifier.

        Returns:
            Verification result.
        """
        return self._client.request("POST", f"/api/v1/audit-trails/{trail_id}/verify")

    # -- Decision receipts (v1) -----------------------------------------------

    def list_receipts(
        self,
        verdict: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List decision receipts.

        Args:
            verdict: Filter by verdict (e.g. APPROVED, REJECTED).
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of decision receipts.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        return self._client.request("GET", "/api/v1/receipts", params=params)

    def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """
        Verify a decision receipt's integrity.

        Args:
            receipt_id: Receipt identifier.

        Returns:
            Verification result.
        """
        return self._client.request("POST", f"/api/v1/receipts/{receipt_id}/verify")


class AsyncAuditTrailAPI:
    """
    Asynchronous Audit Trail API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     trails = await client.audit_trail.list()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # -- Audit trails ---------------------------------------------------------

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List audit trails."""
        params: dict[str, Any] = {"limit": limit, "offset": offset, **kwargs}
        return await self._client.request("GET", "/api/v1/audit-trails", params=params)

    async def get(self, trail_id: str) -> dict[str, Any]:
        """Get an audit trail by ID."""
        return await self._client.request("GET", f"/api/v1/audit-trails/{trail_id}")

    async def export(
        self,
        trail_id: str,
        format: AuditExportFormat = "json",
    ) -> dict[str, Any]:
        """Export an audit trail in the specified format."""
        return await self._client.request(
            "GET",
            f"/api/v1/audit-trails/{trail_id}/export",
            params={"format": format},
        )

    async def verify(self, trail_id: str) -> dict[str, Any]:
        """Verify an audit trail's integrity checksum."""
        return await self._client.request("POST", f"/api/v1/audit-trails/{trail_id}/verify")

    # -- Decision receipts (v1) -----------------------------------------------

    async def list_receipts(
        self,
        verdict: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List decision receipts."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        return await self._client.request("GET", "/api/v1/receipts", params=params)

    async def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Verify a decision receipt's integrity."""
        return await self._client.request("POST", f"/api/v1/receipts/{receipt_id}/verify")


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

    def list(
        self,
        verdict: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List audit trails with pagination.

        Args:
            verdict: Filter by verdict
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of audit trail summaries
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict

        return self._client.request("GET", "/api/v1/audit-trails", params=params)

    def get(self, trail_id: str) -> dict[str, Any]:
        """
        Get a specific audit trail.

        Args:
            trail_id: Audit trail ID

        Returns:
            Full audit trail details
        """
        return self._client.request("GET", f"/api/v1/audit-trails/{trail_id}")

    def export(
        self,
        trail_id: str,
        format: AuditExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export an audit trail.

        Args:
            trail_id: Audit trail ID
            format: Export format (json, csv, md)

        Returns:
            Exported audit trail data
        """
        return self._client.request(
            "GET",
            f"/api/v1/audit-trails/{trail_id}/export",
            params={"format": format},
        )

    def verify(self, trail_id: str) -> dict[str, Any]:
        """
        Verify audit trail integrity.

        Args:
            trail_id: Audit trail ID

        Returns:
            Verification result with checksum comparison
        """
        return self._client.request("POST", f"/api/v1/audit-trails/{trail_id}/verify")

    def list_receipts(
        self,
        verdict: str | None = None,
        risk_level: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List v1 decision receipts.

        Args:
            verdict: Filter by verdict
            risk_level: Filter by risk level
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of receipt summaries
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        if risk_level:
            params["risk_level"] = risk_level

        return self._client.request("GET", "/api/v1/receipts", params=params)

    def get_receipt(self, receipt_id: str) -> dict[str, Any]:
        """
        Get a v1 decision receipt.

        Args:
            receipt_id: Receipt ID

        Returns:
            Receipt details
        """
        return self._client.request("GET", f"/api/v1/receipts/{receipt_id}")

    def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """
        Verify a v1 receipt's integrity.

        Args:
            receipt_id: Receipt ID

        Returns:
            Verification result
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

    async def list(
        self,
        verdict: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List audit trails with pagination."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict

        return await self._client.request("GET", "/api/v1/audit-trails", params=params)

    async def get(self, trail_id: str) -> dict[str, Any]:
        """Get a specific audit trail."""
        return await self._client.request("GET", f"/api/v1/audit-trails/{trail_id}")

    async def export(
        self,
        trail_id: str,
        format: AuditExportFormat = "json",
    ) -> dict[str, Any]:
        """Export an audit trail."""
        return await self._client.request(
            "GET",
            f"/api/v1/audit-trails/{trail_id}/export",
            params={"format": format},
        )

    async def verify(self, trail_id: str) -> dict[str, Any]:
        """Verify audit trail integrity."""
        return await self._client.request("POST", f"/api/v1/audit-trails/{trail_id}/verify")

    async def list_receipts(
        self,
        verdict: str | None = None,
        risk_level: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List v1 decision receipts."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        if risk_level:
            params["risk_level"] = risk_level

        return await self._client.request("GET", "/api/v1/receipts", params=params)

    async def get_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Get a v1 decision receipt."""
        return await self._client.request("GET", f"/api/v1/receipts/{receipt_id}")

    async def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Verify a v1 receipt's integrity."""
        return await self._client.request("POST", f"/api/v1/receipts/{receipt_id}/verify")

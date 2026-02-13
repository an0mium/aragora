"""
Receipts Namespace API

Provides methods for decision receipt management:
- List and retrieve receipts
- Verify receipt integrity
- Export in various formats
- Share receipts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list  # Preserve builtin list for type annotations

ExportFormat = Literal["json", "html", "markdown", "pdf", "sarif", "csv"]


class ReceiptsAPI:
    """
    Synchronous Receipts API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> receipts = client.receipts.list_gauntlet(verdict="PASS")
        >>> for receipt in receipts["results"]:
        ...     print(receipt["receipt_id"], receipt["confidence"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Gauntlet Receipts / Results
    # =========================================================================

    def list_gauntlet(
        self,
        verdict: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List recent gauntlet results (from attack/defend stress tests).

        Args:
            verdict: Filter by verdict
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of gauntlet results
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        return self._client.request("GET", "/api/v2/gauntlet/results", params=params)

    def get_gauntlet(self, receipt_id: str) -> dict[str, Any]:
        """
        Get a gauntlet receipt by gauntlet ID.

        Args:
            receipt_id: Receipt ID

        Returns:
            Gauntlet receipt details
        """
        return self._client.request("GET", f"/api/v2/gauntlet/{receipt_id}/receipt")

    def verify_gauntlet(self, receipt_id: str) -> dict[str, Any]:
        """
        Verify a gauntlet receipt's integrity.

        Args:
            receipt_id: Receipt ID

        Returns:
            Verification result
        """
        return self._client.request(
            "POST",
            f"/api/v2/gauntlet/{receipt_id}/receipt/verify",
        )

    def export_gauntlet(
        self,
        receipt_id: str,
        format: Literal["json", "html", "markdown", "sarif"] = "json",
    ) -> dict[str, Any]:
        """
        Export a gauntlet receipt.

        Args:
            receipt_id: Receipt ID
            format: Export format (json, html, markdown, sarif)

        Returns:
            Exported receipt data
        """
        format_value = "md" if format == "markdown" else format
        return self._client.request(
            "GET",
            f"/api/v2/gauntlet/{receipt_id}/receipt",
            params={"format": format_value},
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def has_dissent(receipt: dict[str, Any]) -> bool:
        """
        Check if a receipt has any dissenting views.

        Args:
            receipt: Receipt data

        Returns:
            True if there are dissenting agents
        """
        dissenting = receipt.get("dissenting_agents", [])
        return len(dissenting) > 0

    @staticmethod
    def get_consensus_status(receipt: dict[str, Any]) -> dict[str, Any]:
        """
        Get the consensus status from a receipt.

        Args:
            receipt: Receipt data

        Returns:
            Consensus status with reached, confidence, and agent counts
        """
        return {
            "reached": receipt.get("consensus_reached", False),
            "confidence": receipt.get("confidence", 0.0),
            "participating_agents": len(receipt.get("participating_agents", [])),
            "dissenting_agents": len(receipt.get("dissenting_agents", [])),
        }


class AsyncReceiptsAPI:
    """
    Asynchronous Receipts API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     results = await client.receipts.list_gauntlet(verdict="PASS")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Gauntlet Receipts / Results
    # =========================================================================

    async def list_gauntlet(
        self,
        verdict: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List recent gauntlet results (from attack/defend stress tests)."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        return await self._client.request("GET", "/api/v2/gauntlet/results", params=params)

    async def get_gauntlet(self, receipt_id: str) -> dict[str, Any]:
        """Get a gauntlet receipt by gauntlet ID."""
        return await self._client.request(
            "GET",
            f"/api/v2/gauntlet/{receipt_id}/receipt",
        )

    async def verify_gauntlet(self, receipt_id: str) -> dict[str, Any]:
        """Verify a gauntlet receipt's integrity."""
        return await self._client.request(
            "POST",
            f"/api/v2/gauntlet/{receipt_id}/receipt/verify",
        )

    async def export_gauntlet(
        self,
        receipt_id: str,
        format: Literal["json", "html", "markdown", "sarif"] = "json",
    ) -> dict[str, Any]:
        """Export a gauntlet receipt."""
        format_value = "md" if format == "markdown" else format
        return await self._client.request(
            "GET",
            f"/api/v2/gauntlet/{receipt_id}/receipt",
            params={"format": format_value},
        )

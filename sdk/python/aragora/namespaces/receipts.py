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

ExportFormat = Literal["json", "html", "markdown", "pdf", "sarif", "csv"]


class ReceiptsAPI:
    """
    Synchronous Receipts API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> receipts = client.receipts.list(verdict="APPROVED")
        >>> for receipt in receipts["receipts"]:
        ...     print(receipt["receipt_id"], receipt["confidence"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(
        self,
        verdict: str | None = None,
        risk_level: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        signed_only: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List decision receipts with filtering.

        Args:
            verdict: Filter by verdict (APPROVED, REJECTED, etc.)
            risk_level: Filter by risk level (low, medium, high, critical)
            from_date: Start date (ISO format)
            to_date: End date (ISO format)
            signed_only: Only return signed receipts
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of receipts with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        if risk_level:
            params["risk_level"] = risk_level
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        if signed_only:
            params["signed_only"] = signed_only

        return self._client.request("GET", "/api/v2/receipts", params=params)

    def get(self, receipt_id: str) -> dict[str, Any]:
        """
        Get a receipt by ID.

        Args:
            receipt_id: Receipt ID

        Returns:
            Receipt details
        """
        return self._client.request("GET", f"/api/v2/receipts/{receipt_id}")

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Full-text search across receipts.

        Args:
            query: Search query (minimum 3 characters)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Matching receipts
        """
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v2/receipts/search", params=params)

    def verify(self, receipt_id: str) -> dict[str, Any]:
        """
        Verify a receipt's integrity.

        Args:
            receipt_id: Receipt ID

        Returns:
            Verification result with checksum and validity
        """
        return self._client.request("POST", f"/api/v2/receipts/{receipt_id}/verify")

    def verify_signature(self, receipt_id: str) -> dict[str, Any]:
        """
        Verify a receipt's cryptographic signature.

        Args:
            receipt_id: Receipt ID

        Returns:
            Signature verification result
        """
        return self._client.request("POST", f"/api/v2/receipts/{receipt_id}/verify-signature")

    def verify_batch(self, receipt_ids: list[str]) -> dict[str, Any]:
        """
        Verify multiple receipts in batch.

        Args:
            receipt_ids: List of receipt IDs (max 100)

        Returns:
            Batch verification results
        """
        return self._client.request(
            "POST",
            "/api/v2/receipts/verify-batch",
            json={"receipt_ids": receipt_ids},
        )

    def export(
        self,
        receipt_id: str,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export a receipt.

        Args:
            receipt_id: Receipt ID
            format: Export format (json, html, markdown, pdf, sarif, csv)

        Returns:
            Exported receipt data or download URL
        """
        return self._client.request(
            "GET",
            f"/api/v2/receipts/{receipt_id}/export",
            params={"format": format},
        )

    def get_stats(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Get receipt statistics.

        Args:
            from_date: Start date
            to_date: End date

        Returns:
            Statistics including counts by verdict, risk level, etc.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return self._client.request("GET", "/api/v2/receipts/stats", params=params)

    def share(
        self,
        receipt_id: str,
        expires_in_hours: int = 24,
        max_accesses: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a shareable link for a receipt.

        Args:
            receipt_id: Receipt ID
            expires_in_hours: Link expiration time
            max_accesses: Maximum number of accesses (None for unlimited)

        Returns:
            Shareable link details
        """
        data: dict[str, Any] = {"expires_in_hours": expires_in_hours}
        if max_accesses is not None:
            data["max_accesses"] = max_accesses

        return self._client.request("POST", f"/api/v2/receipts/{receipt_id}/share", json=data)

    def send_to_channel(
        self,
        receipt_id: str,
        channel_type: str,
        channel_id: str,
    ) -> dict[str, Any]:
        """
        Send a receipt to a channel (Slack, Teams, etc.).

        Args:
            receipt_id: Receipt ID
            channel_type: Channel type (slack, teams, discord, email)
            channel_id: Channel/workspace ID

        Returns:
            Delivery result
        """
        return self._client.request(
            "POST",
            f"/api/v2/receipts/{receipt_id}/send-to-channel",
            json={"channel_type": channel_type, "channel_id": channel_id},
        )

    def get_retention_status(self) -> dict[str, Any]:
        """
        Get retention status for GDPR compliance.

        Returns:
            Retention information including expiry distribution
        """
        return self._client.request("GET", "/api/v2/receipts/retention-status")

    def get_dsar(self, user_id: str) -> dict[str, Any]:
        """
        Get receipts for a user (GDPR Data Subject Access Request).

        Args:
            user_id: User ID

        Returns:
            User's receipts for DSAR compliance
        """
        return self._client.request("GET", f"/api/v2/receipts/dsar/{user_id}")


class AsyncReceiptsAPI:
    """
    Asynchronous Receipts API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     receipts = await client.receipts.list(verdict="APPROVED")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(
        self,
        verdict: str | None = None,
        risk_level: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        signed_only: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List decision receipts with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        if risk_level:
            params["risk_level"] = risk_level
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        if signed_only:
            params["signed_only"] = signed_only

        return await self._client.request("GET", "/api/v2/receipts", params=params)

    async def get(self, receipt_id: str) -> dict[str, Any]:
        """Get a receipt by ID."""
        return await self._client.request("GET", f"/api/v2/receipts/{receipt_id}")

    async def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Full-text search across receipts."""
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v2/receipts/search", params=params)

    async def verify(self, receipt_id: str) -> dict[str, Any]:
        """Verify a receipt's integrity."""
        return await self._client.request("POST", f"/api/v2/receipts/{receipt_id}/verify")

    async def verify_signature(self, receipt_id: str) -> dict[str, Any]:
        """Verify a receipt's cryptographic signature."""
        return await self._client.request("POST", f"/api/v2/receipts/{receipt_id}/verify-signature")

    async def verify_batch(self, receipt_ids: list[str]) -> dict[str, Any]:
        """Verify multiple receipts in batch."""
        return await self._client.request(
            "POST",
            "/api/v2/receipts/verify-batch",
            json={"receipt_ids": receipt_ids},
        )

    async def export(
        self,
        receipt_id: str,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """Export a receipt."""
        return await self._client.request(
            "GET",
            f"/api/v2/receipts/{receipt_id}/export",
            params={"format": format},
        )

    async def get_stats(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Get receipt statistics."""
        params: dict[str, Any] = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return await self._client.request("GET", "/api/v2/receipts/stats", params=params)

    async def share(
        self,
        receipt_id: str,
        expires_in_hours: int = 24,
        max_accesses: int | None = None,
    ) -> dict[str, Any]:
        """Create a shareable link for a receipt."""
        data: dict[str, Any] = {"expires_in_hours": expires_in_hours}
        if max_accesses is not None:
            data["max_accesses"] = max_accesses

        return await self._client.request("POST", f"/api/v2/receipts/{receipt_id}/share", json=data)

    async def send_to_channel(
        self,
        receipt_id: str,
        channel_type: str,
        channel_id: str,
    ) -> dict[str, Any]:
        """Send a receipt to a channel."""
        return await self._client.request(
            "POST",
            f"/api/v2/receipts/{receipt_id}/send-to-channel",
            json={"channel_type": channel_type, "channel_id": channel_id},
        )

    async def get_retention_status(self) -> dict[str, Any]:
        """Get retention status for GDPR compliance."""
        return await self._client.request("GET", "/api/v2/receipts/retention-status")

    async def get_dsar(self, user_id: str) -> dict[str, Any]:
        """Get receipts for a user (GDPR DSAR)."""
        return await self._client.request("GET", f"/api/v2/receipts/dsar/{user_id}")

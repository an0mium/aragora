"""
Gauntlet Namespace API

Provides methods for the Gauntlet adversarial validation system:
- Run attack/defend cycles
- Manage decision receipts
- Access vulnerability findings
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ExportFormat = Literal["json", "html", "markdown", "pdf", "sarif", "csv"]


class GauntletAPI:
    """
    Synchronous Gauntlet API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.gauntlet.run(debate_id="dbt_123")
        >>> print(result["verdict"], result["confidence"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def run(
        self,
        debate_id: str | None = None,
        task: str | None = None,
        attack_rounds: int = 3,
        proposer_agent: str | None = None,
        attacker_agents: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run a Gauntlet adversarial validation.

        Args:
            debate_id: Existing debate to validate (optional)
            task: New task to validate (if no debate_id)
            attack_rounds: Number of attack/defend cycles
            proposer_agent: Agent to defend the position
            attacker_agents: Agents to attack the position

        Returns:
            Gauntlet result with verdict, findings, and receipt
        """
        data: dict[str, Any] = {"attack_rounds": attack_rounds}
        if debate_id:
            data["debate_id"] = debate_id
        if task:
            data["task"] = task
        if proposer_agent:
            data["proposer_agent"] = proposer_agent
        if attacker_agents:
            data["attacker_agents"] = attacker_agents

        return self._client.request("POST", "/api/v1/gauntlet/run", json=data)

    def get_result(self, gauntlet_id: str) -> dict[str, Any]:
        """
        Get a Gauntlet result by ID.

        Args:
            gauntlet_id: Gauntlet run ID

        Returns:
            Gauntlet result details
        """
        return self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}")

    def get_receipt(self, gauntlet_id: str) -> dict[str, Any]:
        """
        Get the decision receipt for a Gauntlet run.

        Args:
            gauntlet_id: Gauntlet run ID

        Returns:
            Decision receipt with cryptographic proof
        """
        return self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/receipt")

    def verify_receipt(self, gauntlet_id: str) -> dict[str, Any]:
        """
        Verify a Gauntlet receipt's integrity.

        Args:
            gauntlet_id: Gauntlet run ID

        Returns:
            Verification result with valid status and hash
        """
        return self._client.request("POST", f"/api/v1/gauntlet/{gauntlet_id}/receipt/verify")

    def compare_receipts(self, receipt_id: str, other_id: str) -> dict[str, Any]:
        """Compare two gauntlet runs side-by-side."""
        return self._client.request(
            "GET", f"/api/v1/gauntlet/{receipt_id}/compare/{other_id}"
        )

    def run_and_wait(
        self,
        timeout: float = 300,
        poll_interval: float = 2.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a Gauntlet validation and wait for the result.

        Convenience method that calls ``run()`` then polls ``get_result()``
        until a terminal state is reached.

        Args:
            timeout: Maximum seconds to wait
            poll_interval: Seconds between status polls
            **kwargs: Arguments forwarded to ``run()``

        Returns:
            The completed gauntlet result
        """
        import time

        result = self.run(**kwargs)
        gauntlet_id = result.get("gauntlet_id") or result.get("id", "")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_result(gauntlet_id)
            if status.get("status") in ("completed", "failed"):
                return status
            time.sleep(poll_interval)
        return self.get_result(gauntlet_id)

    def list(self) -> dict[str, Any]:
        """List gauntlet runs."""
        return self._client.request("GET", "/api/v1/gauntlet")

    def list_heatmaps(self) -> dict[str, Any]:
        """List gauntlet heatmaps."""
        return self._client.request("GET", "/api/v1/gauntlet/heatmaps")

    def get_heatmap(self, heatmap_id: str) -> dict[str, Any]:
        """Get a heatmap."""
        return self._client.request("GET", f"/api/v1/gauntlet/heatmaps/{heatmap_id}")

    def export_heatmap(self, heatmap_id: str) -> dict[str, Any]:
        """Export a heatmap."""
        return self._client.request("GET", f"/api/v1/gauntlet/heatmaps/{heatmap_id}/export")

    def list_receipts(self) -> dict[str, Any]:
        """List receipts."""
        return self._client.request("GET", "/api/v1/gauntlet/receipts")

    def export_receipt_bundle(self) -> dict[str, Any]:
        """Export receipt bundle."""
        return self._client.request("POST", "/api/v1/gauntlet/receipts/export/bundle")

    def get_receipt_by_id(self, receipt_id: str) -> dict[str, Any]:
        """Get a receipt by ID."""
        return self._client.request("GET", f"/api/v1/gauntlet/receipts/{receipt_id}")

    def export_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Export a receipt."""
        return self._client.request("GET", f"/api/v1/gauntlet/receipts/{receipt_id}/export")

    def stream_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Stream a receipt."""
        return self._client.request("GET", f"/api/v1/gauntlet/receipts/{receipt_id}/stream")

    def compare(self, gauntlet_id: str) -> dict[str, Any]:
        """Compare gauntlet runs."""
        return self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/compare")


class AsyncGauntletAPI:
    """
    Asynchronous Gauntlet API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.gauntlet.run(task="Should we deploy to prod?")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def run(
        self,
        debate_id: str | None = None,
        task: str | None = None,
        attack_rounds: int = 3,
        proposer_agent: str | None = None,
        attacker_agents: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a Gauntlet adversarial validation."""
        data: dict[str, Any] = {"attack_rounds": attack_rounds}
        if debate_id:
            data["debate_id"] = debate_id
        if task:
            data["task"] = task
        if proposer_agent:
            data["proposer_agent"] = proposer_agent
        if attacker_agents:
            data["attacker_agents"] = attacker_agents

        return await self._client.request("POST", "/api/v1/gauntlet/run", json=data)

    async def get_result(self, gauntlet_id: str) -> dict[str, Any]:
        """Get a Gauntlet result by ID."""
        return await self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}")

    async def get_receipt(self, gauntlet_id: str) -> dict[str, Any]:
        """Get the decision receipt for a Gauntlet run."""
        return await self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/receipt")

    async def verify_receipt(self, gauntlet_id: str) -> dict[str, Any]:
        """Verify a Gauntlet receipt's integrity."""
        return await self._client.request("POST", f"/api/v1/gauntlet/{gauntlet_id}/receipt/verify")

    async def compare_receipts(self, receipt_id: str, other_id: str) -> dict[str, Any]:
        """Compare two gauntlet runs side-by-side."""
        return await self._client.request(
            "GET", f"/api/v1/gauntlet/{receipt_id}/compare/{other_id}"
        )

    async def run_and_wait(
        self,
        timeout: float = 300,
        poll_interval: float = 2.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a Gauntlet validation and wait for the result.

        Async convenience method that calls ``run()`` then polls
        ``get_result()`` until a terminal state is reached.

        Args:
            timeout: Maximum seconds to wait
            poll_interval: Seconds between status polls
            **kwargs: Arguments forwarded to ``run()``

        Returns:
            The completed gauntlet result
        """
        import asyncio

        result = await self.run(**kwargs)
        gauntlet_id = result.get("gauntlet_id") or result.get("id", "")
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            status = await self.get_result(gauntlet_id)
            if status.get("status") in ("completed", "failed"):
                return status
            await asyncio.sleep(poll_interval)
        return await self.get_result(gauntlet_id)

    async def list(self) -> dict[str, Any]:
        """List gauntlet runs."""
        return await self._client.request("GET", "/api/v1/gauntlet")

    async def list_heatmaps(self) -> dict[str, Any]:
        """List gauntlet heatmaps."""
        return await self._client.request("GET", "/api/v1/gauntlet/heatmaps")

    async def get_heatmap(self, heatmap_id: str) -> dict[str, Any]:
        """Get a heatmap."""
        return await self._client.request("GET", f"/api/v1/gauntlet/heatmaps/{heatmap_id}")

    async def export_heatmap(self, heatmap_id: str) -> dict[str, Any]:
        """Export a heatmap."""
        return await self._client.request("GET", f"/api/v1/gauntlet/heatmaps/{heatmap_id}/export")

    async def list_receipts(self) -> dict[str, Any]:
        """List receipts."""
        return await self._client.request("GET", "/api/v1/gauntlet/receipts")

    async def export_receipt_bundle(self) -> dict[str, Any]:
        """Export receipt bundle."""
        return await self._client.request("POST", "/api/v1/gauntlet/receipts/export/bundle")

    async def get_receipt_by_id(self, receipt_id: str) -> dict[str, Any]:
        """Get a receipt by ID."""
        return await self._client.request("GET", f"/api/v1/gauntlet/receipts/{receipt_id}")

    async def export_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Export a receipt."""
        return await self._client.request("GET", f"/api/v1/gauntlet/receipts/{receipt_id}/export")

    async def stream_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Stream a receipt."""
        return await self._client.request("GET", f"/api/v1/gauntlet/receipts/{receipt_id}/stream")

    async def compare(self, gauntlet_id: str) -> dict[str, Any]:
        """Compare gauntlet runs."""
        return await self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/compare")

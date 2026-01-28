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

    def list_results(
        self,
        verdict: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List Gauntlet results with filtering.

        Args:
            verdict: Filter by verdict (PASS, CONDITIONAL, FAIL)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of Gauntlet results
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict

        return self._client.request("GET", "/api/v1/gauntlet", params=params)

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

    def export_receipt(
        self,
        gauntlet_id: str,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export a Gauntlet receipt.

        Args:
            gauntlet_id: Gauntlet run ID
            format: Export format (json, html, markdown, pdf, sarif, csv)

        Returns:
            Exported receipt data or download URL
        """
        return self._client.request(
            "GET",
            f"/api/v1/gauntlet/{gauntlet_id}/receipt/export",
            params={"format": format},
        )

    def get_findings(self, gauntlet_id: str) -> dict[str, Any]:
        """
        Get findings from a Gauntlet run.

        Args:
            gauntlet_id: Gauntlet run ID

        Returns:
            List of findings with severity and details
        """
        return self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/findings")

    def get_attacks(self, gauntlet_id: str) -> dict[str, Any]:
        """
        Get attack details from a Gauntlet run.

        Args:
            gauntlet_id: Gauntlet run ID

        Returns:
            Attack rounds with arguments and outcomes
        """
        return self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/attacks")

    def get_stats(self) -> dict[str, Any]:
        """
        Get Gauntlet statistics.

        Returns:
            Statistics including pass rates, common findings, etc.
        """
        return self._client.request("GET", "/api/v1/gauntlet/stats")


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

    async def list_results(
        self,
        verdict: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List Gauntlet results with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict

        return await self._client.request("GET", "/api/v1/gauntlet", params=params)

    async def get_receipt(self, gauntlet_id: str) -> dict[str, Any]:
        """Get the decision receipt for a Gauntlet run."""
        return await self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/receipt")

    async def verify_receipt(self, gauntlet_id: str) -> dict[str, Any]:
        """Verify a Gauntlet receipt's integrity."""
        return await self._client.request("POST", f"/api/v1/gauntlet/{gauntlet_id}/receipt/verify")

    async def export_receipt(
        self,
        gauntlet_id: str,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """Export a Gauntlet receipt."""
        return await self._client.request(
            "GET",
            f"/api/v1/gauntlet/{gauntlet_id}/receipt/export",
            params={"format": format},
        )

    async def get_findings(self, gauntlet_id: str) -> dict[str, Any]:
        """Get findings from a Gauntlet run."""
        return await self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/findings")

    async def get_attacks(self, gauntlet_id: str) -> dict[str, Any]:
        """Get attack details from a Gauntlet run."""
        return await self._client.request("GET", f"/api/v1/gauntlet/{gauntlet_id}/attacks")

    async def get_stats(self) -> dict[str, Any]:
        """Get Gauntlet statistics."""
        return await self._client.request("GET", "/api/v1/gauntlet/stats")

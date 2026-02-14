"""
Auditing Namespace API

Provides deep audit, capability probing, and red team analysis capabilities:
- Probe agent capabilities for specific tasks
- Run deep audits on tasks
- Perform red team analysis on debates
- Get available attack types
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

AuditDepth = Literal["shallow", "standard", "deep"]
AttackIntensity = Literal["low", "medium", "high"]
Severity = Literal["low", "medium", "high", "critical"]

class AuditingAPI:
    """
    Synchronous Auditing API.

    Provides deep audit, capability probing, and red team analysis.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.auditing.probe_capability(
        ...     agent="claude",
        ...     task="Code review for security vulnerabilities"
        ... )
        >>> print(result["score"], result["capabilities"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def probe_capability(
        self,
        agent: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run capability probes on an agent.

        Args:
            agent: The agent identifier to probe
            task: The task to test the agent's capabilities on
            config: Optional configuration for the probe

        Returns:
            Capability probe result including score, capabilities,
            weaknesses, and recommendations
        """
        data: dict[str, Any] = {"agent": agent, "task": task}
        if config:
            data["config"] = config

        return self._client.request("POST", "/api/v1/debates/capability-probe", json=data)

    def deep_audit(
        self,
        task: str,
        agents: list[str] | None = None,
        depth: AuditDepth = "standard",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run a deep audit on a task.

        Args:
            task: The task to audit
            agents: Optional list of agents to include in the audit
            depth: Audit depth (shallow, standard, deep)
            config: Optional configuration for the audit

        Returns:
            Deep audit result including audit_id, findings, and summary
        """
        data: dict[str, Any] = {"task": task, "depth": depth}
        if agents:
            data["agents"] = agents
        if config:
            data["config"] = config

        return self._client.request("POST", "/api/v1/debates/deep-audit", json=data)

    def get_attack_types(self) -> dict[str, Any]:
        """
        Get available red team attack types.

        Returns:
            Dictionary containing list of available attack types
            with their id, name, description, and category
        """
        return self._client.request("GET", "/api/v1/redteam/attack-types")

class AsyncAuditingAPI:
    """
    Asynchronous Auditing API.

    Provides deep audit, capability probing, and red team analysis.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.auditing.probe_capability(
        ...         agent="claude",
        ...         task="Code review for security vulnerabilities"
        ...     )
        ...     print(result["score"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def probe_capability(
        self,
        agent: str,
        task: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run capability probes on an agent.

        Args:
            agent: The agent identifier to probe
            task: The task to test the agent's capabilities on
            config: Optional configuration for the probe

        Returns:
            Capability probe result including score, capabilities,
            weaknesses, and recommendations
        """
        data: dict[str, Any] = {"agent": agent, "task": task}
        if config:
            data["config"] = config

        return await self._client.request("POST", "/api/v1/debates/capability-probe", json=data)

    async def deep_audit(
        self,
        task: str,
        agents: list[str] | None = None,
        depth: AuditDepth = "standard",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run a deep audit on a task.

        Args:
            task: The task to audit
            agents: Optional list of agents to include in the audit
            depth: Audit depth (shallow, standard, deep)
            config: Optional configuration for the audit

        Returns:
            Deep audit result including audit_id, findings, and summary
        """
        data: dict[str, Any] = {"task": task, "depth": depth}
        if agents:
            data["agents"] = agents
        if config:
            data["config"] = config

        return await self._client.request("POST", "/api/v1/debates/deep-audit", json=data)

    async def get_attack_types(self) -> dict[str, Any]:
        """
        Get available red team attack types.

        Returns:
            Dictionary containing list of available attack types
            with their id, name, description, and category
        """
        return await self._client.request("GET", "/api/v1/redteam/attack-types")

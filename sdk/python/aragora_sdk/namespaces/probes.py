"""
Probes Namespace API

Provides access to agent capability probing:
- Probe individual agent capabilities
- Run comprehensive probe suites
- View probe reports and results
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ProbesAPI:
    """
    Synchronous Probes API for agent capability testing.

    Provides methods for probing agent capabilities, running comprehensive
    probe suites, and viewing probe reports.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.probes.probe_capability(
        ...     agent="claude", capability="reasoning"
        ... )
        >>> report = client.probes.run(agents=["claude", "gpt-4"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def probe_capability(
        self,
        agent: str,
        capability: str,
        test_input: str | None = None,
    ) -> dict[str, Any]:
        """
        Probe an agent's specific capability.

        Args:
            agent: Agent to probe.
            capability: Capability to test (e.g., 'reasoning', 'coding',
                'analysis', 'creativity').
            test_input: Optional test input to use for the probe.

        Returns:
            Dict with probe results including:
            - score: Capability score (0.0-1.0)
            - latency_ms: Response latency
            - assessment: Detailed assessment
        """
        body: dict[str, Any] = {
            "agent": agent,
            "capability": capability,
        }
        if test_input:
            body["test_input"] = test_input
        return self._client.request("POST", "/api/v1/probes/capability", json=body)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Run a comprehensive probe suite against one or more agents.

        Args:
            **kwargs: Probe suite parameters including:
                - agents: List of agents to probe
                - capabilities: List of capabilities to test (default: all)
                - iterations: Number of probe iterations per capability
                - timeout_ms: Maximum time per probe

        Returns:
            Dict with probe suite results including per-agent
            and per-capability scores.
        """
        return self._client.request("POST", "/api/v1/probes/run", json=kwargs)

    def get_reports(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get probe reports.

        Args:
            **kwargs: Report filter parameters including:
                - agent: Filter by agent name
                - capability: Filter by capability
                - limit: Maximum reports to return

        Returns:
            Dict with probe reports and historical performance data.
        """
        return self._client.request("POST", "/api/v1/probes/reports", json=kwargs)


class AsyncProbesAPI:
    """
    Asynchronous Probes API for agent capability testing.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.probes.probe_capability(
        ...         agent="claude", capability="reasoning"
        ...     )
        ...     report = await client.probes.run(agents=["claude", "gpt-4"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def probe_capability(
        self,
        agent: str,
        capability: str,
        test_input: str | None = None,
    ) -> dict[str, Any]:
        """
        Probe an agent's specific capability.

        Args:
            agent: Agent to probe.
            capability: Capability to test.
            test_input: Optional test input.

        Returns:
            Dict with probe results.
        """
        body: dict[str, Any] = {
            "agent": agent,
            "capability": capability,
        }
        if test_input:
            body["test_input"] = test_input
        return await self._client.request("POST", "/api/v1/probes/capability", json=body)

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run a comprehensive probe suite against one or more agents."""
        return await self._client.request("POST", "/api/v1/probes/run", json=kwargs)

    async def get_reports(self, **kwargs: Any) -> dict[str, Any]:
        """Get probe reports."""
        return await self._client.request("POST", "/api/v1/probes/reports", json=kwargs)

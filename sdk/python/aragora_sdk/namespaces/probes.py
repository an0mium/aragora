"""
Probes Namespace API

Provides access to agent capability probing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ProbesAPI:
    """Synchronous Probes API for agent capability testing."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def probe_capability(
        self,
        agent: str,
        capability: str,
        test_input: str | None = None,
    ) -> dict[str, Any]:
        """Probe an agent's capability.

        Args:
            agent: Agent to probe.
            capability: Capability to test.
            test_input: Optional test input.

        Returns:
            Probe results with capability assessment.
        """
        body: dict[str, Any] = {
            "agent": agent,
            "capability": capability,
        }
        if test_input:
            body["test_input"] = test_input
        return self._client.request("POST", "/api/v1/probes/capability", json=body)


class AsyncProbesAPI:
    """Asynchronous Probes API for agent capability testing."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def probe_capability(
        self,
        agent: str,
        capability: str,
        test_input: str | None = None,
    ) -> dict[str, Any]:
        """Probe an agent's capability.

        Args:
            agent: Agent to probe.
            capability: Capability to test.
            test_input: Optional test input.

        Returns:
            Probe results with capability assessment.
        """
        body: dict[str, Any] = {
            "agent": agent,
            "capability": capability,
        }
        if test_input:
            body["test_input"] = test_input
        return await self._client.request("POST", "/api/v1/probes/capability", json=body)

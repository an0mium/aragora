"""
Debates Namespace API

Provides methods for creating, managing, and analyzing debates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class DebatesAPI:
    """
    Synchronous Debates API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> debate = client.debates.create(task="Should we use microservices?")
        >>> messages = client.debates.get_messages(debate["debate_id"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def create(
        self,
        task: str,
        agents: list[str] | None = None,
        protocol: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create a new debate.

        Args:
            task: The topic or question to debate
            agents: List of agent names to participate (optional)
            protocol: Debate protocol configuration (optional)
            **kwargs: Additional debate options

        Returns:
            Created debate with debate_id
        """
        data = {"task": task, **kwargs}
        if agents:
            data["agents"] = agents
        if protocol:
            data["protocol"] = protocol

        return self._client.request("POST", "/api/v1/debates", json=data)

    def get(self, debate_id: str) -> dict[str, Any]:
        """
        Get a debate by ID.

        Args:
            debate_id: The debate ID

        Returns:
            Debate details
        """
        return self._client.request("GET", f"/api/v1/debates/{debate_id}")

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List debates with pagination.

        Args:
            limit: Maximum number of debates to return
            offset: Number of debates to skip
            status: Filter by status (active, completed, etc.)

        Returns:
            List of debates with pagination info
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v1/debates", params=params)

    def get_messages(self, debate_id: str) -> dict[str, Any]:
        """
        Get messages from a debate.

        Args:
            debate_id: The debate ID

        Returns:
            List of messages
        """
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/messages")

    def add_message(
        self,
        debate_id: str,
        content: str,
        role: str = "user",
    ) -> dict[str, Any]:
        """
        Add a message to a debate.

        Args:
            debate_id: The debate ID
            content: Message content
            role: Message role (user, system, etc.)

        Returns:
            Created message
        """
        return self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/messages",
            json={"content": content, "role": role},
        )

    def get_consensus(self, debate_id: str) -> dict[str, Any]:
        """
        Get consensus information for a debate.

        Args:
            debate_id: The debate ID

        Returns:
            Consensus details
        """
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/consensus")

    def get_export(
        self,
        debate_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """
        Export a debate.

        Args:
            debate_id: The debate ID
            format: Export format (json, pdf, etc.)

        Returns:
            Exported debate data
        """
        return self._client.request(
            "GET",
            f"/api/v1/debates/{debate_id}/export",
            params={"format": format},
        )

    def cancel(self, debate_id: str) -> dict[str, Any]:
        """
        Cancel a debate.

        Args:
            debate_id: The debate ID

        Returns:
            Cancellation result
        """
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/cancel")


class AsyncDebatesAPI:
    """
    Asynchronous Debates API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     debate = await client.debates.create(task="Should we use microservices?")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create(
        self,
        task: str,
        agents: list[str] | None = None,
        protocol: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a new debate."""
        data = {"task": task, **kwargs}
        if agents:
            data["agents"] = agents
        if protocol:
            data["protocol"] = protocol

        return await self._client.request("POST", "/api/v1/debates", json=data)

    async def get(self, debate_id: str) -> dict[str, Any]:
        """Get a debate by ID."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}")

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List debates with pagination."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v1/debates", params=params)

    async def get_messages(self, debate_id: str) -> dict[str, Any]:
        """Get messages from a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/messages")

    async def add_message(
        self,
        debate_id: str,
        content: str,
        role: str = "user",
    ) -> dict[str, Any]:
        """Add a message to a debate."""
        return await self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/messages",
            json={"content": content, "role": role},
        )

    async def get_consensus(self, debate_id: str) -> dict[str, Any]:
        """Get consensus information for a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/consensus")

    async def get_export(
        self,
        debate_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Export a debate."""
        return await self._client.request(
            "GET",
            f"/api/v1/debates/{debate_id}/export",
            params={"format": format},
        )

    async def cancel(self, debate_id: str) -> dict[str, Any]:
        """Cancel a debate."""
        return await self._client.request("POST", f"/api/v1/debates/{debate_id}/cancel")

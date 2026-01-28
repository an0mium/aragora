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

    def get_by_slug(self, slug: str) -> dict[str, Any]:
        """Get a debate by slug."""
        return self._client.request("GET", f"/api/v1/debates/slug/{slug}")

    def get_explainability(self, debate_id: str) -> dict[str, Any]:
        """Get explainability data for a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability")

    def get_explainability_factors(self, debate_id: str) -> dict[str, Any]:
        """Get factor decomposition for a debate decision."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability/factors")

    def get_explainability_narrative(self, debate_id: str) -> dict[str, Any]:
        """Get natural language narrative explanation."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability/narrative")

    def get_explainability_provenance(self, debate_id: str) -> dict[str, Any]:
        """Get provenance chain for debate claims."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability/provenance")

    def get_explainability_counterfactual(self, debate_id: str) -> dict[str, Any]:
        """Get counterfactual analysis."""
        return self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/counterfactual"
        )

    def create_counterfactual(self, debate_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Create a counterfactual scenario."""
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/explainability/counterfactual", json=changes
        )

    def get_citations(self, debate_id: str) -> dict[str, Any]:
        """Get citations used in debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/citations")

    def get_convergence(self, debate_id: str) -> dict[str, Any]:
        """Get convergence analysis."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/convergence")

    def get_evidence(self, debate_id: str) -> dict[str, Any]:
        """Get evidence collected during debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/evidence")

    def get_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get graph statistics for debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/graph/stats")

    def get_impasse(self, debate_id: str) -> dict[str, Any]:
        """Get impasse analysis if debate is stuck."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/impasse")

    def get_meta_critique(self, debate_id: str) -> dict[str, Any]:
        """Get meta-critique of debate process."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/meta-critique")

    def get_red_team(self, debate_id: str) -> dict[str, Any]:
        """Get red team analysis."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/red-team")

    def capability_probe(self, task: str, agents: list[str] | None = None) -> dict[str, Any]:
        """Run a capability probe debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        return self._client.request("POST", "/api/v1/debates/capability-probe", json=data)

    def deep_audit(self, task: str, agents: list[str] | None = None) -> dict[str, Any]:
        """Run a deep audit debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        return self._client.request("POST", "/api/v1/debates/deep-audit", json=data)

    def broadcast(self, debate_id: str, channels: list[str] | None = None) -> dict[str, Any]:
        """Broadcast debate to channels."""
        data: dict[str, Any] = {"channels": channels} if channels else {}
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/broadcast", json=data)

    def fork(self, debate_id: str, changes: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fork a debate with optional changes."""
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/fork", json=changes or {})

    def publish_twitter(self, debate_id: str, message: str | None = None) -> dict[str, Any]:
        """Publish debate summary to Twitter."""
        data: dict[str, Any] = {"message": message} if message else {}
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/publish/twitter", json=data
        )

    def publish_youtube(self, debate_id: str, title: str | None = None) -> dict[str, Any]:
        """Publish debate to YouTube."""
        data: dict[str, Any] = {"title": title} if title else {}
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/publish/youtube", json=data
        )

    def get_dashboard(self) -> dict[str, Any]:
        """Get debates dashboard view."""
        return self._client.request("GET", "/api/v1/dashboard/debates")

    def get_debate_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get graph stats via debate endpoint."""
        return self._client.request("GET", f"/api/v1/debate/{debate_id}/graph-stats")

    def get_history(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """Get debate history."""
        return self._client.request(
            "GET", "/api/v1/history/debates", params={"limit": limit, "offset": offset}
        )


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

    async def get_by_slug(self, slug: str) -> dict[str, Any]:
        """Get a debate by slug."""
        return await self._client.request("GET", f"/api/v1/debates/slug/{slug}")

    async def get_explainability(self, debate_id: str) -> dict[str, Any]:
        """Get explainability data for a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability")

    async def get_explainability_factors(self, debate_id: str) -> dict[str, Any]:
        """Get factor decomposition for a debate decision."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/factors"
        )

    async def get_explainability_narrative(self, debate_id: str) -> dict[str, Any]:
        """Get natural language narrative explanation."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/narrative"
        )

    async def get_explainability_provenance(self, debate_id: str) -> dict[str, Any]:
        """Get provenance chain for debate claims."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/provenance"
        )

    async def get_explainability_counterfactual(self, debate_id: str) -> dict[str, Any]:
        """Get counterfactual analysis."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/counterfactual"
        )

    async def create_counterfactual(
        self, debate_id: str, changes: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a counterfactual scenario."""
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/explainability/counterfactual", json=changes
        )

    async def get_citations(self, debate_id: str) -> dict[str, Any]:
        """Get citations used in debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/citations")

    async def get_convergence(self, debate_id: str) -> dict[str, Any]:
        """Get convergence analysis."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/convergence")

    async def get_evidence(self, debate_id: str) -> dict[str, Any]:
        """Get evidence collected during debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/evidence")

    async def get_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get graph statistics for debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/graph/stats")

    async def get_impasse(self, debate_id: str) -> dict[str, Any]:
        """Get impasse analysis if debate is stuck."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/impasse")

    async def get_meta_critique(self, debate_id: str) -> dict[str, Any]:
        """Get meta-critique of debate process."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/meta-critique")

    async def get_red_team(self, debate_id: str) -> dict[str, Any]:
        """Get red team analysis."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/red-team")

    async def capability_probe(self, task: str, agents: list[str] | None = None) -> dict[str, Any]:
        """Run a capability probe debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        return await self._client.request("POST", "/api/v1/debates/capability-probe", json=data)

    async def deep_audit(self, task: str, agents: list[str] | None = None) -> dict[str, Any]:
        """Run a deep audit debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        return await self._client.request("POST", "/api/v1/debates/deep-audit", json=data)

    async def broadcast(self, debate_id: str, channels: list[str] | None = None) -> dict[str, Any]:
        """Broadcast debate to channels."""
        data: dict[str, Any] = {"channels": channels} if channels else {}
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/broadcast", json=data
        )

    async def fork(self, debate_id: str, changes: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fork a debate with optional changes."""
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/fork", json=changes or {}
        )

    async def publish_twitter(self, debate_id: str, message: str | None = None) -> dict[str, Any]:
        """Publish debate summary to Twitter."""
        data: dict[str, Any] = {"message": message} if message else {}
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/publish/twitter", json=data
        )

    async def publish_youtube(self, debate_id: str, title: str | None = None) -> dict[str, Any]:
        """Publish debate to YouTube."""
        data: dict[str, Any] = {"title": title} if title else {}
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/publish/youtube", json=data
        )

    async def get_dashboard(self) -> dict[str, Any]:
        """Get debates dashboard view."""
        return await self._client.request("GET", "/api/v1/dashboard/debates")

    async def get_debate_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get graph stats via debate endpoint."""
        return await self._client.request("GET", f"/api/v1/debate/{debate_id}/graph-stats")

    async def get_history(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """Get debate history."""
        return await self._client.request(
            "GET", "/api/v1/history/debates", params={"limit": limit, "offset": offset}
        )

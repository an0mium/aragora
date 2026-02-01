"""
Orchestration Namespace API.

Provides methods for unified multi-agent deliberation orchestration:
- Async and sync deliberation endpoints
- Knowledge context integration
- Output channel routing
- Template-based workflows

Endpoints:
    POST /api/v1/orchestration/deliberate      - Async deliberation
    POST /api/v1/orchestration/deliberate/sync - Sync deliberation
    GET  /api/v1/orchestration/status/:id      - Get status
    GET  /api/v1/orchestration/templates       - List templates
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

TeamStrategy = Literal["specified", "best_for_domain", "diverse", "fast", "random"]
OutputFormat = Literal["standard", "decision_receipt", "summary", "github_review", "slack_message"]


class OrchestrationAPI:
    """
    Synchronous Orchestration API.

    Provides methods for unified multi-agent deliberation:
    - Submit deliberations (async or sync)
    - Check deliberation status
    - List available templates

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Async deliberation
        >>> result = client.orchestration.deliberate(
        ...     question="Should we migrate to Kubernetes?",
        ...     knowledge_sources=["confluence:12345", "slack:C123456"],
        ...     output_channels=["slack:C789"],
        ... )
        >>> # Check status
        >>> status = client.orchestration.get_status(result["request_id"])
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Deliberation
    # =========================================================================

    def deliberate(
        self,
        question: str,
        knowledge_sources: list[str | dict[str, Any]] | None = None,
        workspaces: list[str] | None = None,
        team_strategy: TeamStrategy = "best_for_domain",
        agents: list[str] | None = None,
        output_channels: list[str | dict[str, Any]] | None = None,
        output_format: OutputFormat = "standard",
        require_consensus: bool = True,
        priority: str = "normal",
        max_rounds: int = 3,
        timeout_seconds: float = 300.0,
        template: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit an async deliberation request.

        Returns immediately with a request_id that can be used to check status.

        Args:
            question: The question or decision to deliberate on.
            knowledge_sources: Sources for context (e.g., "slack:C123", "confluence:12345").
            workspaces: Workspace IDs to include in context.
            team_strategy: Strategy for agent team selection.
            agents: Explicit list of agents to use.
            output_channels: Channels to route results to (e.g., "slack:C789").
            output_format: Format for the output.
            require_consensus: Whether consensus is required.
            priority: Priority level (low, normal, high, critical).
            max_rounds: Maximum deliberation rounds.
            timeout_seconds: Timeout for the deliberation.
            template: Template name to use.
            metadata: Additional metadata.

        Returns:
            Dict with request_id and status.
        """
        data: dict[str, Any] = {"question": question}

        if knowledge_sources:
            data["knowledge_sources"] = knowledge_sources
        if workspaces:
            data["workspaces"] = workspaces
        if team_strategy != "best_for_domain":
            data["team_strategy"] = team_strategy
        if agents:
            data["agents"] = agents
        if output_channels:
            data["output_channels"] = output_channels
        if output_format != "standard":
            data["output_format"] = output_format
        if not require_consensus:
            data["require_consensus"] = require_consensus
        if priority != "normal":
            data["priority"] = priority
        if max_rounds != 3:
            data["max_rounds"] = max_rounds
        if timeout_seconds != 300.0:
            data["timeout_seconds"] = timeout_seconds
        if template:
            data["template"] = template
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/v1/orchestration/deliberate", json=data)

    def deliberate_sync(
        self,
        question: str,
        knowledge_sources: list[str | dict[str, Any]] | None = None,
        workspaces: list[str] | None = None,
        team_strategy: TeamStrategy = "best_for_domain",
        agents: list[str] | None = None,
        output_channels: list[str | dict[str, Any]] | None = None,
        output_format: OutputFormat = "standard",
        require_consensus: bool = True,
        priority: str = "normal",
        max_rounds: int = 3,
        timeout_seconds: float = 300.0,
        template: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a synchronous deliberation request.

        Blocks until deliberation completes and returns the full result.

        Args:
            question: The question or decision to deliberate on.
            knowledge_sources: Sources for context.
            workspaces: Workspace IDs to include.
            team_strategy: Strategy for agent team selection.
            agents: Explicit list of agents to use.
            output_channels: Channels to route results to.
            output_format: Format for the output.
            require_consensus: Whether consensus is required.
            priority: Priority level.
            max_rounds: Maximum deliberation rounds.
            timeout_seconds: Timeout for the deliberation.
            template: Template name to use.
            metadata: Additional metadata.

        Returns:
            Full deliberation result.
        """
        data: dict[str, Any] = {"question": question}

        if knowledge_sources:
            data["knowledge_sources"] = knowledge_sources
        if workspaces:
            data["workspaces"] = workspaces
        if team_strategy != "best_for_domain":
            data["team_strategy"] = team_strategy
        if agents:
            data["agents"] = agents
        if output_channels:
            data["output_channels"] = output_channels
        if output_format != "standard":
            data["output_format"] = output_format
        if not require_consensus:
            data["require_consensus"] = require_consensus
        if priority != "normal":
            data["priority"] = priority
        if max_rounds != 3:
            data["max_rounds"] = max_rounds
        if timeout_seconds != 300.0:
            data["timeout_seconds"] = timeout_seconds
        if template:
            data["template"] = template
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/v1/orchestration/deliberate/sync", json=data)

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self, request_id: str) -> dict[str, Any]:
        """
        Get the status of a deliberation request.

        Args:
            request_id: The deliberation request ID.

        Returns:
            Dict with status and result (if completed).
        """
        return self._client.request("GET", f"/api/v1/orchestration/status/{request_id}")

    # =========================================================================
    # Templates
    # =========================================================================

    def list_templates(self) -> dict[str, Any]:
        """
        List available deliberation templates.

        Returns:
            Dict with templates list and count.
        """
        return self._client.request("GET", "/api/v1/orchestration/templates")


class AsyncOrchestrationAPI:
    """Asynchronous Orchestration API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Deliberation
    # =========================================================================

    async def deliberate(
        self,
        question: str,
        knowledge_sources: list[str | dict[str, Any]] | None = None,
        workspaces: list[str] | None = None,
        team_strategy: TeamStrategy = "best_for_domain",
        agents: list[str] | None = None,
        output_channels: list[str | dict[str, Any]] | None = None,
        output_format: OutputFormat = "standard",
        require_consensus: bool = True,
        priority: str = "normal",
        max_rounds: int = 3,
        timeout_seconds: float = 300.0,
        template: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit an async deliberation request."""
        data: dict[str, Any] = {"question": question}

        if knowledge_sources:
            data["knowledge_sources"] = knowledge_sources
        if workspaces:
            data["workspaces"] = workspaces
        if team_strategy != "best_for_domain":
            data["team_strategy"] = team_strategy
        if agents:
            data["agents"] = agents
        if output_channels:
            data["output_channels"] = output_channels
        if output_format != "standard":
            data["output_format"] = output_format
        if not require_consensus:
            data["require_consensus"] = require_consensus
        if priority != "normal":
            data["priority"] = priority
        if max_rounds != 3:
            data["max_rounds"] = max_rounds
        if timeout_seconds != 300.0:
            data["timeout_seconds"] = timeout_seconds
        if template:
            data["template"] = template
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("POST", "/api/v1/orchestration/deliberate", json=data)

    async def deliberate_sync(
        self,
        question: str,
        knowledge_sources: list[str | dict[str, Any]] | None = None,
        workspaces: list[str] | None = None,
        team_strategy: TeamStrategy = "best_for_domain",
        agents: list[str] | None = None,
        output_channels: list[str | dict[str, Any]] | None = None,
        output_format: OutputFormat = "standard",
        require_consensus: bool = True,
        priority: str = "normal",
        max_rounds: int = 3,
        timeout_seconds: float = 300.0,
        template: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a synchronous deliberation request."""
        data: dict[str, Any] = {"question": question}

        if knowledge_sources:
            data["knowledge_sources"] = knowledge_sources
        if workspaces:
            data["workspaces"] = workspaces
        if team_strategy != "best_for_domain":
            data["team_strategy"] = team_strategy
        if agents:
            data["agents"] = agents
        if output_channels:
            data["output_channels"] = output_channels
        if output_format != "standard":
            data["output_format"] = output_format
        if not require_consensus:
            data["require_consensus"] = require_consensus
        if priority != "normal":
            data["priority"] = priority
        if max_rounds != 3:
            data["max_rounds"] = max_rounds
        if timeout_seconds != 300.0:
            data["timeout_seconds"] = timeout_seconds
        if template:
            data["template"] = template
        if metadata:
            data["metadata"] = metadata

        return await self._client.request(
            "POST", "/api/v1/orchestration/deliberate/sync", json=data
        )

    # =========================================================================
    # Status
    # =========================================================================

    async def get_status(self, request_id: str) -> dict[str, Any]:
        """Get the status of a deliberation request."""
        return await self._client.request("GET", f"/api/v1/orchestration/status/{request_id}")

    # =========================================================================
    # Templates
    # =========================================================================

    async def list_templates(self) -> dict[str, Any]:
        """List available deliberation templates."""
        return await self._client.request("GET", "/api/v1/orchestration/templates")

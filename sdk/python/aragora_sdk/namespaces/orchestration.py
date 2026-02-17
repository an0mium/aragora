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
        >>> result = client.orchestration.deliberate(
        ...     question="Should we migrate to Kubernetes?",
        ...     knowledge_sources=["confluence:12345", "slack:C123456"],
        ...     output_channels=["slack:C789"],
        ... )
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
        knowledge_sources: list[str] | None = None,
        output_channels: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit an asynchronous deliberation request."""
        body: dict[str, Any] = {"question": question, **kwargs}
        if knowledge_sources:
            body["knowledge_sources"] = knowledge_sources
        if output_channels:
            body["output_channels"] = output_channels
        return self._client.request("POST", "/api/v1/orchestration/deliberate", json=body)

    def deliberate_sync(
        self,
        question: str,
        knowledge_sources: list[str] | None = None,
        output_channels: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit a synchronous deliberation request (blocks until complete)."""
        body: dict[str, Any] = {"question": question, **kwargs}
        if knowledge_sources:
            body["knowledge_sources"] = knowledge_sources
        if output_channels:
            body["output_channels"] = output_channels
        return self._client.request("POST", "/api/v1/orchestration/deliberate/sync", json=body)

    def get_status(self, request_id: str) -> dict[str, Any]:
        """Get the status of a deliberation request."""
        return self._client.request("GET", f"/api/v1/orchestration/status/{request_id}")

    def list_templates(self) -> dict[str, Any]:
        """List available orchestration templates."""
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
        knowledge_sources: list[str] | None = None,
        output_channels: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit an asynchronous deliberation request."""
        body: dict[str, Any] = {"question": question, **kwargs}
        if knowledge_sources:
            body["knowledge_sources"] = knowledge_sources
        if output_channels:
            body["output_channels"] = output_channels
        return await self._client.request("POST", "/api/v1/orchestration/deliberate", json=body)

    async def deliberate_sync(
        self,
        question: str,
        knowledge_sources: list[str] | None = None,
        output_channels: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit a synchronous deliberation request."""
        body: dict[str, Any] = {"question": question, **kwargs}
        if knowledge_sources:
            body["knowledge_sources"] = knowledge_sources
        if output_channels:
            body["output_channels"] = output_channels
        return await self._client.request("POST", "/api/v1/orchestration/deliberate/sync", json=body)

    async def get_status(self, request_id: str) -> dict[str, Any]:
        """Get the status of a deliberation request."""
        return await self._client.request("GET", f"/api/v1/orchestration/status/{request_id}")

    async def list_templates(self) -> dict[str, Any]:
        """List available orchestration templates."""
        return await self._client.request("GET", "/api/v1/orchestration/templates")

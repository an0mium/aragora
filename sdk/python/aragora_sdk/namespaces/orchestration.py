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

from typing import TYPE_CHECKING, Literal

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

class AsyncOrchestrationAPI:
    """Asynchronous Orchestration API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Deliberation


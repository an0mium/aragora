"""Graph debates API resource for branching debate structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.client.client import AragoraClient

from aragora.client.models import (
    GraphDebate,
    GraphDebateBranch,
    GraphDebateCreateRequest,
    GraphDebateCreateResponse,
)


class GraphDebatesAPI:
    """API interface for graph-structured debates with branching."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def create(
        self,
        task: str,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        branch_threshold: float = 0.5,
        max_branches: int = 5,
    ) -> GraphDebateCreateResponse:
        """
        Create and start a graph-structured debate.

        Graph debates allow for automatic branching when agents
        identify fundamentally different approaches.

        Args:
            task: The question or topic to debate.
            agents: List of agent IDs to participate.
            max_rounds: Maximum rounds per branch (1-20).
            branch_threshold: Divergence threshold for branching (0-1).
            max_branches: Maximum number of branches allowed.

        Returns:
            GraphDebateCreateResponse with debate_id.
        """
        request = GraphDebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            max_rounds=max_rounds,
            branch_threshold=branch_threshold,
            max_branches=max_branches,
        )

        response = self._client._post("/api/debates/graph", request.model_dump())
        return GraphDebateCreateResponse(**response)

    async def create_async(
        self,
        task: str,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        branch_threshold: float = 0.5,
        max_branches: int = 5,
    ) -> GraphDebateCreateResponse:
        """Async version of create()."""
        request = GraphDebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            max_rounds=max_rounds,
            branch_threshold=branch_threshold,
            max_branches=max_branches,
        )

        response = await self._client._post_async("/api/debates/graph", request.model_dump())
        return GraphDebateCreateResponse(**response)

    def get(self, debate_id: str) -> GraphDebate:
        """
        Get graph debate details by ID.

        Args:
            debate_id: The graph debate ID.

        Returns:
            GraphDebate with full details including branches.
        """
        response = self._client._get(f"/api/debates/graph/{debate_id}")
        return GraphDebate(**response)

    async def get_async(self, debate_id: str) -> GraphDebate:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/debates/graph/{debate_id}")
        return GraphDebate(**response)

    def get_branches(self, debate_id: str) -> list[GraphDebateBranch]:
        """
        Get all branches for a graph debate.

        Args:
            debate_id: The graph debate ID.

        Returns:
            List of GraphDebateBranch objects.
        """
        response = self._client._get(f"/api/debates/graph/{debate_id}/branches")
        branches = response.get("branches", response) if isinstance(response, dict) else response
        return [GraphDebateBranch(**b) for b in branches]

    async def get_branches_async(self, debate_id: str) -> list[GraphDebateBranch]:
        """Async version of get_branches()."""
        response = await self._client._get_async(f"/api/debates/graph/{debate_id}/branches")
        branches = response.get("branches", response) if isinstance(response, dict) else response
        return [GraphDebateBranch(**b) for b in branches]


__all__ = ["GraphDebatesAPI"]

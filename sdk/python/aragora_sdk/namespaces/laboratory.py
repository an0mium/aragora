"""
Laboratory Namespace API.

Provides access to the persona laboratory for detecting emergent traits
and suggesting beneficial cross-pollinations between agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class LaboratoryAPI:
    """
    Synchronous Laboratory API.

    Provides access to persona laboratory features for detecting
    emergent traits and suggesting beneficial cross-pollinations.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> traits = client.laboratory.get_emergent_traits(min_confidence=0.8)
        >>> suggestions = client.laboratory.suggest_cross_pollinations("claude")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def get_emergent_traits(
        self,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get emergent traits detected from agent performance.

        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0).
            limit: Maximum number of traits to return.

        Returns:
            List of emergent traits with confidence scores and evidence.
        """
        params: dict[str, Any] = {}
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        if limit is not None:
            params["limit"] = limit

        return self._client._request("GET", "/api/v2/laboratory/emergent-traits", params=params)

    def suggest_cross_pollinations(
        self,
        target_agent: str,
    ) -> dict[str, Any]:
        """
        Suggest beneficial trait transfers for a target agent.

        Args:
            target_agent: Agent to receive trait suggestions.

        Returns:
            Cross-pollination suggestions with source agents and reasons.
        """
        return self._client._request(
            "POST",
            "/api/v2/laboratory/cross-pollinations",
            json={"target_agent": target_agent},
        )

    def get_trait_analysis(
        self,
        agent_name: str,
    ) -> dict[str, Any]:
        """
        Get detailed trait analysis for an agent.

        Args:
            agent_name: Name of the agent to analyze.

        Returns:
            Trait analysis with strengths, weaknesses, and recommendations.
        """
        return self._client._request("GET", f"/api/v2/laboratory/agent/{agent_name}/analysis")

    def run_experiment(
        self,
        experiment_type: str,
        agents: list[str],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run a laboratory experiment.

        Args:
            experiment_type: Type of experiment (trait_transfer, comparison, etc.).
            agents: List of agents to include in experiment.
            config: Optional experiment configuration.

        Returns:
            Experiment results with metrics and observations.
        """
        data: dict[str, Any] = {
            "experiment_type": experiment_type,
            "agents": agents,
        }
        if config:
            data["config"] = config

        return self._client._request("POST", "/api/v2/laboratory/experiments", json=data)

    def get_experiments(
        self,
        limit: int | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List laboratory experiments.

        Args:
            limit: Maximum number of experiments to return.
            status: Filter by status (pending, running, completed).

        Returns:
            List of experiments with status and results.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if status is not None:
            params["status"] = status

        return self._client._request("GET", "/api/v2/laboratory/experiments", params=params)


class AsyncLaboratoryAPI:
    """Asynchronous Laboratory API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def get_emergent_traits(
        self,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Get emergent traits detected from agent performance."""
        params: dict[str, Any] = {}
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        if limit is not None:
            params["limit"] = limit

        return await self._client._request(
            "GET", "/api/v2/laboratory/emergent-traits", params=params
        )

    async def suggest_cross_pollinations(
        self,
        target_agent: str,
    ) -> dict[str, Any]:
        """Suggest beneficial trait transfers for a target agent."""
        return await self._client._request(
            "POST",
            "/api/v2/laboratory/cross-pollinations",
            json={"target_agent": target_agent},
        )

    async def get_trait_analysis(
        self,
        agent_name: str,
    ) -> dict[str, Any]:
        """Get detailed trait analysis for an agent."""
        return await self._client._request("GET", f"/api/v2/laboratory/agent/{agent_name}/analysis")

    async def run_experiment(
        self,
        experiment_type: str,
        agents: list[str],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a laboratory experiment."""
        data: dict[str, Any] = {
            "experiment_type": experiment_type,
            "agents": agents,
        }
        if config:
            data["config"] = config

        return await self._client._request("POST", "/api/v2/laboratory/experiments", json=data)

    async def get_experiments(
        self,
        limit: int | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List laboratory experiments."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if status is not None:
            params["status"] = status

        return await self._client._request("GET", "/api/v2/laboratory/experiments", params=params)

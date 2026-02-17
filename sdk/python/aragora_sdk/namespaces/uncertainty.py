"""
Uncertainty Namespace API

Provides uncertainty estimation, agent calibration profiles, and follow-up generation:
- Estimate uncertainty for content or debate responses
- Get uncertainty metrics for debates
- Get calibration profiles for agents
- Generate follow-up suggestions from uncertainty cruxes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class UncertaintyAPI:
    """
    Synchronous Uncertainty API.

    Provides uncertainty estimation, agent calibration profiles, and follow-up generation.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> estimate = client.uncertainty.estimate(
        ...     content="The market will grow by 15% next quarter",
        ...     context="Financial analysis"
        ... )
        >>> print(estimate["overall_uncertainty"])
        >>> print(estimate["cruxes"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def estimate(
        self,
        content: str,
        context: str | None = None,
        debate_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Estimate uncertainty for content or a debate response.

        Args:
            content: The content to analyze for uncertainty
            context: Optional context for the content
            debate_id: Optional debate ID for additional context
            config: Optional configuration for the estimation

        Returns:
            Uncertainty estimate including:
            - overall_uncertainty: float between 0-1
            - confidence_interval: tuple of (low, high)
            - cruxes: list of key uncertainties with importance and uncertainty scores
            - methodology: description of estimation method
        """
        data: dict[str, Any] = {"content": content}
        if context:
            data["context"] = context
        if debate_id:
            data["debate_id"] = debate_id
        if config:
            data["config"] = config

        return self._client.request("POST", "/api/v1/uncertainty/estimate", json=data)

    def list_debate_uncertainties(self) -> dict[str, Any]:
        """
        List uncertainty summaries across debates.

        Returns:
            List of debate uncertainty summaries.
        """
        return self._client.request("POST", "/api/v1/uncertainty/debate")

    def get_debate_metrics(self, debate_id: str) -> dict[str, Any]:
        """
        Get uncertainty metrics for a debate.

        Args:
            debate_id: The debate ID

        Returns:
            Debate uncertainty metrics including:
            - debate_id: The debate identifier
            - overall_uncertainty: Aggregate uncertainty score
            - round_uncertainties: List of uncertainty by round
            - agent_uncertainties: Dict of uncertainty by agent
            - convergence_trend: How uncertainty changed over rounds
            - cruxes: Key points of disagreement/uncertainty
        """
        return self._client.request("GET", f"/api/v1/uncertainty/debate/{debate_id}")

    def get_agent_profile(self, agent_id: str) -> dict[str, Any]:
        """
        Get calibration profile for an agent.

        Args:
            agent_id: The agent identifier

        Returns:
            Agent calibration profile including:
            - agent_id: The agent identifier
            - calibration_score: Overall calibration score (0-1)
            - overconfidence_bias: Tendency to be overconfident
            - accuracy_by_confidence: Breakdown of accuracy per confidence level
            - total_predictions: Number of predictions made
        """
        return self._client.request("GET", f"/api/v1/uncertainty/agent/{agent_id}")

    def generate_followups(
        self,
        debate_id: str | None = None,
        cruxes: list[dict[str, Any]] | None = None,
        max_suggestions: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate follow-up suggestions from uncertainty cruxes.

        Args:
            debate_id: Optional debate ID to generate follow-ups for
            cruxes: Optional list of cruxes with description and importance
            max_suggestions: Maximum number of suggestions to generate

        Returns:
            Dictionary with 'suggestions' list, each containing:
            - question: Suggested follow-up question
            - rationale: Why this follow-up would help
            - priority: Importance score
            - related_crux: Which crux this addresses
        """
        data: dict[str, Any] = {}
        if debate_id:
            data["debate_id"] = debate_id
        if cruxes:
            data["cruxes"] = cruxes
        if max_suggestions:
            data["max_suggestions"] = max_suggestions

        return self._client.request("POST", "/api/v1/uncertainty/followups", json=data)


class AsyncUncertaintyAPI:
    """
    Asynchronous Uncertainty API.

    Provides uncertainty estimation, agent calibration profiles, and follow-up generation.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     estimate = await client.uncertainty.estimate(
        ...         content="The market will grow by 15% next quarter",
        ...         context="Financial analysis"
        ...     )
        ...     print(estimate["overall_uncertainty"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def estimate(
        self,
        content: str,
        context: str | None = None,
        debate_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Estimate uncertainty for content or a debate response.

        Args:
            content: The content to analyze for uncertainty
            context: Optional context for the content
            debate_id: Optional debate ID for additional context
            config: Optional configuration for the estimation

        Returns:
            Uncertainty estimate including:
            - overall_uncertainty: float between 0-1
            - confidence_interval: tuple of (low, high)
            - cruxes: list of key uncertainties with importance and uncertainty scores
            - methodology: description of estimation method
        """
        data: dict[str, Any] = {"content": content}
        if context:
            data["context"] = context
        if debate_id:
            data["debate_id"] = debate_id
        if config:
            data["config"] = config

        return await self._client.request("POST", "/api/v1/uncertainty/estimate", json=data)

    async def list_debate_uncertainties(self) -> dict[str, Any]:
        """List uncertainty summaries across debates."""
        return await self._client.request("POST", "/api/v1/uncertainty/debate")

    async def get_debate_metrics(self, debate_id: str) -> dict[str, Any]:
        """
        Get uncertainty metrics for a debate.

        Args:
            debate_id: The debate ID

        Returns:
            Debate uncertainty metrics including:
            - debate_id: The debate identifier
            - overall_uncertainty: Aggregate uncertainty score
            - round_uncertainties: List of uncertainty by round
            - agent_uncertainties: Dict of uncertainty by agent
            - convergence_trend: How uncertainty changed over rounds
            - cruxes: Key points of disagreement/uncertainty
        """
        return await self._client.request("GET", f"/api/v1/uncertainty/debate/{debate_id}")

    async def get_agent_profile(self, agent_id: str) -> dict[str, Any]:
        """
        Get calibration profile for an agent.

        Args:
            agent_id: The agent identifier

        Returns:
            Agent calibration profile including:
            - agent_id: The agent identifier
            - calibration_score: Overall calibration score (0-1)
            - overconfidence_bias: Tendency to be overconfident
            - accuracy_by_confidence: Breakdown of accuracy per confidence level
            - total_predictions: Number of predictions made
        """
        return await self._client.request("GET", f"/api/v1/uncertainty/agent/{agent_id}")

    async def generate_followups(
        self,
        debate_id: str | None = None,
        cruxes: list[dict[str, Any]] | None = None,
        max_suggestions: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate follow-up suggestions from uncertainty cruxes.

        Args:
            debate_id: Optional debate ID to generate follow-ups for
            cruxes: Optional list of cruxes with description and importance
            max_suggestions: Maximum number of suggestions to generate

        Returns:
            Dictionary with 'suggestions' list, each containing:
            - question: Suggested follow-up question
            - rationale: Why this follow-up would help
            - priority: Importance score
            - related_crux: Which crux this addresses
        """
        data: dict[str, Any] = {}
        if debate_id:
            data["debate_id"] = debate_id
        if cruxes:
            data["cruxes"] = cruxes
        if max_suggestions:
            data["max_suggestions"] = max_suggestions

        return await self._client.request("POST", "/api/v1/uncertainty/followups", json=data)

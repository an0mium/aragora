"""
Explainability Namespace API

Provides methods for decision explainability:
- Get natural language explanations
- Access factor decomposition
- Generate counterfactual analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ExplainabilityAPI:
    """
    Synchronous Explainability API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> explanation = client.explainability.explain("decision_123")
        >>> print(explanation["summary"])
        >>> for factor in explanation["factors"]:
        ...     print(factor["name"], factor["weight"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def explain(
        self,
        decision_id: str,
        audience: str = "technical",
        include_factors: bool = True,
        include_evidence: bool = True,
    ) -> dict[str, Any]:
        """
        Get a natural language explanation for a decision.

        Args:
            decision_id: Decision/debate ID
            audience: Target audience (technical, executive, compliance)
            include_factors: Include factor breakdown
            include_evidence: Include supporting evidence

        Returns:
            Explanation with summary, factors, and evidence
        """
        params: dict[str, Any] = {
            "audience": audience,
            "include_factors": include_factors,
            "include_evidence": include_evidence,
        }
        return self._client.request(
            "GET", f"/api/v1/explainability/decisions/{decision_id}", params=params
        )

    def get_factors(
        self,
        decision_id: str,
        min_weight: float | None = None,
    ) -> dict[str, Any]:
        """
        Get factor decomposition for a decision.

        Args:
            decision_id: Decision/debate ID
            min_weight: Minimum factor weight to include

        Returns:
            List of factors with weights and descriptions
        """
        params: dict[str, Any] = {}
        if min_weight is not None:
            params["min_weight"] = min_weight

        return self._client.request(
            "GET",
            f"/api/v1/explainability/decisions/{decision_id}/factors",
            params=params,
        )

    def get_counterfactuals(
        self,
        decision_id: str,
        target_outcome: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """
        Get counterfactual analysis for a decision.

        Args:
            decision_id: Decision/debate ID
            target_outcome: Target outcome to analyze (e.g., "APPROVED")
            limit: Maximum counterfactuals to return

        Returns:
            Counterfactual scenarios showing what would change the outcome
        """
        params: dict[str, Any] = {"limit": limit}
        if target_outcome:
            params["target_outcome"] = target_outcome

        return self._client.request(
            "GET",
            f"/api/v1/explainability/decisions/{decision_id}/counterfactuals",
            params=params,
        )

    def generate_summary(
        self,
        decision_id: str,
        format: str = "paragraph",
        max_length: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate a natural language summary.

        Args:
            decision_id: Decision/debate ID
            format: Summary format (paragraph, bullets, executive)
            max_length: Maximum summary length in words

        Returns:
            Generated summary
        """
        params: dict[str, Any] = {"format": format}
        if max_length is not None:
            params["max_length"] = max_length

        return self._client.request(
            "GET",
            f"/api/v1/explainability/decisions/{decision_id}/summary",
            params=params,
        )

    def get_evidence_chain(self, decision_id: str) -> dict[str, Any]:
        """
        Get the evidence chain supporting a decision.

        Args:
            decision_id: Decision/debate ID

        Returns:
            Evidence chain with sources and citations
        """
        return self._client.request(
            "GET", f"/api/v1/explainability/decisions/{decision_id}/evidence"
        )

    def get_agent_contributions(self, decision_id: str) -> dict[str, Any]:
        """
        Get contribution breakdown by agent.

        Args:
            decision_id: Decision/debate ID

        Returns:
            Agent contributions with impact scores
        """
        return self._client.request(
            "GET", f"/api/v1/explainability/decisions/{decision_id}/contributions"
        )

    def get_confidence_breakdown(self, decision_id: str) -> dict[str, Any]:
        """
        Get confidence score breakdown.

        Args:
            decision_id: Decision/debate ID

        Returns:
            Confidence breakdown by source and factor
        """
        return self._client.request(
            "GET", f"/api/v1/explainability/decisions/{decision_id}/confidence"
        )

    def compare_decisions(
        self,
        decision_id_1: str,
        decision_id_2: str,
    ) -> dict[str, Any]:
        """
        Compare two decisions.

        Args:
            decision_id_1: First decision ID
            decision_id_2: Second decision ID

        Returns:
            Comparison analysis highlighting differences
        """
        return self._client.request(
            "GET",
            "/api/v1/explainability/compare",
            params={"decision_1": decision_id_1, "decision_2": decision_id_2},
        )


class AsyncExplainabilityAPI:
    """
    Asynchronous Explainability API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     explanation = await client.explainability.explain("decision_123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def explain(
        self,
        decision_id: str,
        audience: str = "technical",
        include_factors: bool = True,
        include_evidence: bool = True,
    ) -> dict[str, Any]:
        """Get a natural language explanation for a decision."""
        params: dict[str, Any] = {
            "audience": audience,
            "include_factors": include_factors,
            "include_evidence": include_evidence,
        }
        return await self._client.request(
            "GET", f"/api/v1/explainability/decisions/{decision_id}", params=params
        )

    async def get_factors(
        self,
        decision_id: str,
        min_weight: float | None = None,
    ) -> dict[str, Any]:
        """Get factor decomposition for a decision."""
        params: dict[str, Any] = {}
        if min_weight is not None:
            params["min_weight"] = min_weight

        return await self._client.request(
            "GET",
            f"/api/v1/explainability/decisions/{decision_id}/factors",
            params=params,
        )

    async def get_counterfactuals(
        self,
        decision_id: str,
        target_outcome: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Get counterfactual analysis for a decision."""
        params: dict[str, Any] = {"limit": limit}
        if target_outcome:
            params["target_outcome"] = target_outcome

        return await self._client.request(
            "GET",
            f"/api/v1/explainability/decisions/{decision_id}/counterfactuals",
            params=params,
        )

    async def generate_summary(
        self,
        decision_id: str,
        format: str = "paragraph",
        max_length: int | None = None,
    ) -> dict[str, Any]:
        """Generate a natural language summary."""
        params: dict[str, Any] = {"format": format}
        if max_length is not None:
            params["max_length"] = max_length

        return await self._client.request(
            "GET",
            f"/api/v1/explainability/decisions/{decision_id}/summary",
            params=params,
        )

    async def get_evidence_chain(self, decision_id: str) -> dict[str, Any]:
        """Get the evidence chain supporting a decision."""
        return await self._client.request(
            "GET", f"/api/v1/explainability/decisions/{decision_id}/evidence"
        )

    async def get_agent_contributions(self, decision_id: str) -> dict[str, Any]:
        """Get contribution breakdown by agent."""
        return await self._client.request(
            "GET", f"/api/v1/explainability/decisions/{decision_id}/contributions"
        )

    async def get_confidence_breakdown(self, decision_id: str) -> dict[str, Any]:
        """Get confidence score breakdown."""
        return await self._client.request(
            "GET", f"/api/v1/explainability/decisions/{decision_id}/confidence"
        )

    async def compare_decisions(
        self,
        decision_id_1: str,
        decision_id_2: str,
    ) -> dict[str, Any]:
        """Compare two decisions."""
        return await self._client.request(
            "GET",
            "/api/v1/explainability/compare",
            params={"decision_1": decision_id_1, "decision_2": decision_id_2},
        )

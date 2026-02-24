"""
Explainability Namespace API

Provides methods for decision explainability:
- Get natural language explanations for decisions
- Access factor decomposition
- Generate counterfactual analysis
- Compare decisions
- Batch explanation processing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ExplainabilityAPI:
    """
    Synchronous Explainability API.

    Provides natural language explanations for decisions, factor decomposition,
    counterfactual analysis, and batch processing.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> explanation = client.explainability.get_explanation("decision_123")
        >>> print(explanation["summary"])
        >>> batch = client.explainability.batch_explain(
        ...     decision_ids=["d1", "d2", "d3"]
        ... )
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Explanations
    # =========================================================================

    def list_explanations(self) -> dict[str, Any]:
        """
        List available explanations.

        Returns:
            Dict with list of available decision explanations.
        """
        return self._client.request("GET", "/api/v1/explain")

    def get_explanation(self, decision_id: str) -> dict[str, Any]:
        """
        Get a full explanation for a decision or debate.

        Args:
            decision_id: Decision or debate identifier.

        Returns:
            Dict with explanation including:
            - summary: Natural language summary
            - factors: Factor decomposition
            - confidence: Confidence scores
            - counterfactuals: What-if analysis
        """
        return self._client.request("GET", f"/api/v1/explain/{decision_id}")

    # =========================================================================
    # Comparison
    # =========================================================================

    def compare_decisions(
        self,
        decision_id_1: str,
        decision_id_2: str,
    ) -> dict[str, Any]:
        """
        Compare two decisions side-by-side.

        Args:
            decision_id_1: First decision ID.
            decision_id_2: Second decision ID.

        Returns:
            Dict with comparison analysis highlighting differences
            in reasoning, factors, and outcomes.
        """
        return self._client.request(
            "GET",
            "/api/v1/explainability/compare",
            params={"decision_1": decision_id_1, "decision_2": decision_id_2},
        )

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def batch_explain(self, **kwargs: Any) -> dict[str, Any]:
        """
        Submit a batch of decisions for explanation.

        Args:
            **kwargs: Batch parameters including:
                - decision_ids: List of decision IDs to explain
                - detail_level: Level of detail (summary, standard, detailed)

        Returns:
            Dict with batch job ID and status. Use get_batch_status()
            to check progress and get_batch_results() to retrieve results.
        """
        return self._client.request("POST", "/api/v1/explainability/batch", json=kwargs)

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """
        Get status of a batch explanation job.

        Args:
            batch_id: Batch job identifier.

        Returns:
            Dict with batch status including progress percentage
            and completion state.
        """
        return self._client.request("GET", f"/api/v1/explainability/batch/{batch_id}/status")

    def get_batch_results(self, batch_id: str) -> dict[str, Any]:
        """
        Get results of a completed batch explanation job.

        Args:
            batch_id: Batch job identifier.

        Returns:
            Dict with explanation results for each decision in the batch.
        """
        return self._client.request("GET", f"/api/v1/explainability/batch/{batch_id}/results")


class AsyncExplainabilityAPI:
    """
    Asynchronous Explainability API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     explanation = await client.explainability.get_explanation("decision_123")
        ...     batch = await client.explainability.batch_explain(
        ...         decision_ids=["d1", "d2"]
        ...     )
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Explanations
    # =========================================================================

    async def list_explanations(self) -> dict[str, Any]:
        """List available explanations."""
        return await self._client.request("GET", "/api/v1/explain")

    async def get_explanation(self, decision_id: str) -> dict[str, Any]:
        """Get a full explanation for a decision or debate."""
        return await self._client.request("GET", f"/api/v1/explain/{decision_id}")

    # =========================================================================
    # Comparison
    # =========================================================================

    async def compare_decisions(
        self,
        decision_id_1: str,
        decision_id_2: str,
    ) -> dict[str, Any]:
        """Compare two decisions side-by-side."""
        return await self._client.request(
            "GET",
            "/api/v1/explainability/compare",
            params={"decision_1": decision_id_1, "decision_2": decision_id_2},
        )

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def batch_explain(self, **kwargs: Any) -> dict[str, Any]:
        """Submit a batch of decisions for explanation."""
        return await self._client.request("POST", "/api/v1/explainability/batch", json=kwargs)

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Get status of a batch explanation job."""
        return await self._client.request("GET", f"/api/v1/explainability/batch/{batch_id}/status")

    async def get_batch_results(self, batch_id: str) -> dict[str, Any]:
        """Get results of a completed batch explanation job."""
        return await self._client.request("GET", f"/api/v1/explainability/batch/{batch_id}/results")

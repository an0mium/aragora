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
        >>> explanation = client.explainability.get_explanation("decision_123")
        >>> print(explanation["summary"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_explanations(self) -> dict[str, Any]:
        """List available explanations."""
        return self._client.request("GET", "/api/v1/explain")

    def get_explanation(self, decision_id: str) -> dict[str, Any]:
        """Get a full explanation for a decision or debate."""
        return self._client.request("GET", f"/api/v1/explain/{decision_id}")

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
        ...     explanation = await client.explainability.get_explanation("decision_123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_explanations(self) -> dict[str, Any]:
        """List available explanations."""
        return await self._client.request("GET", "/api/v1/explain")

    async def get_explanation(self, decision_id: str) -> dict[str, Any]:
        """Get a full explanation for a decision or debate."""
        return await self._client.request("GET", f"/api/v1/explain/{decision_id}")

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

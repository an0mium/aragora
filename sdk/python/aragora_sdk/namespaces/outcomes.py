"""
Outcomes namespace for decision outcome tracking.

Provides API methods for recording real-world outcomes of decisions,
searching past outcomes, and viewing impact analytics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class OutcomesAPI:
    """Synchronous outcomes API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def record(
        self,
        decision_id: str,
        debate_id: str,
        outcome_type: str,
        outcome_description: str,
        impact_score: float,
        kpis_before: dict[str, Any] | None = None,
        kpis_after: dict[str, Any] | None = None,
        lessons_learned: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Record an outcome for a decision.

        Args:
            decision_id: The decision this outcome relates to
            debate_id: The debate that produced the decision
            outcome_type: One of success, failure, partial, unknown
            outcome_description: Description of what happened
            impact_score: Impact score from 0.0 to 1.0
            kpis_before: KPI values before the decision was executed
            kpis_after: KPI values after the decision was executed
            lessons_learned: What was learned from this outcome
            tags: Optional tags for categorization

        Returns:
            Outcome record with outcome_id
        """
        data: dict[str, Any] = {
            "debate_id": debate_id,
            "outcome_type": outcome_type,
            "outcome_description": outcome_description,
            "impact_score": impact_score,
        }
        if kpis_before:
            data["kpis_before"] = kpis_before
        if kpis_after:
            data["kpis_after"] = kpis_after
        if lessons_learned:
            data["lessons_learned"] = lessons_learned
        if tags:
            data["tags"] = tags

        return self._client._request(
            "POST", f"/api/v1/decisions/{decision_id}/outcome", json=data
        )

    def list(self, decision_id: str) -> dict[str, Any]:
        """
        List outcomes for a decision.

        Args:
            decision_id: Decision identifier

        Returns:
            List of outcomes with count
        """
        return self._client._request("GET", f"/api/v1/decisions/{decision_id}/outcomes")

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        outcome_type: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Search outcomes by topic, tags, or type.

        Args:
            query: Text search query
            tags: Filter by tags
            outcome_type: Filter by type (success/failure/partial/unknown)
            limit: Maximum results

        Returns:
            Matching outcomes
        """
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        if tags:
            params["tags"] = ",".join(tags)
        if outcome_type:
            params["type"] = outcome_type

        return self._client._request("GET", "/api/v1/outcomes/search", params=params)

    def impact(self) -> dict[str, Any]:
        """
        Get impact analytics across all outcomes.

        Returns:
            Aggregate statistics grouped by outcome type
        """
        return self._client._request("GET", "/api/v1/outcomes/impact")


class AsyncOutcomesAPI:
    """Asynchronous outcomes API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def record(
        self,
        decision_id: str,
        debate_id: str,
        outcome_type: str,
        outcome_description: str,
        impact_score: float,
        kpis_before: dict[str, Any] | None = None,
        kpis_after: dict[str, Any] | None = None,
        lessons_learned: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record an outcome for a decision."""
        data: dict[str, Any] = {
            "debate_id": debate_id,
            "outcome_type": outcome_type,
            "outcome_description": outcome_description,
            "impact_score": impact_score,
        }
        if kpis_before:
            data["kpis_before"] = kpis_before
        if kpis_after:
            data["kpis_after"] = kpis_after
        if lessons_learned:
            data["lessons_learned"] = lessons_learned
        if tags:
            data["tags"] = tags

        return await self._client._request(
            "POST", f"/api/v1/decisions/{decision_id}/outcome", json=data
        )

    async def list(self, decision_id: str) -> dict[str, Any]:
        """List outcomes for a decision."""
        return await self._client._request(
            "GET", f"/api/v1/decisions/{decision_id}/outcomes"
        )

    async def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        outcome_type: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search outcomes by topic, tags, or type."""
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        if tags:
            params["tags"] = ",".join(tags)
        if outcome_type:
            params["type"] = outcome_type

        return await self._client._request(
            "GET", "/api/v1/outcomes/search", params=params
        )

    async def impact(self) -> dict[str, Any]:
        """Get impact analytics across all outcomes."""
        return await self._client._request("GET", "/api/v1/outcomes/impact")

"""Explainability API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class ExplanationFactor(BaseModel):
    """Factor contributing to a decision."""

    name: str
    contribution: float
    description: str | None = None
    evidence: list[str] | None = None


class Explanation(BaseModel):
    """Decision explanation."""

    debate_id: str
    summary: str
    factors: list[ExplanationFactor] | None = None
    confidence: float | None = None
    methodology: str | None = None
    created_at: str | None = None


class Counterfactual(BaseModel):
    """Counterfactual scenario."""

    id: str
    scenario: str
    outcome: str
    probability: float | None = None
    key_changes: list[str] | None = None


class ProvenanceEntry(BaseModel):
    """Provenance tracking entry."""

    source: str
    timestamp: str
    action: str
    agent: str | None = None
    details: dict[str, Any] | None = None


class Narrative(BaseModel):
    """Decision narrative."""

    debate_id: str
    format: str
    content: str
    word_count: int | None = None
    created_at: str | None = None


class BatchExplanationJob(BaseModel):
    """Batch explanation job status."""

    batch_id: str
    status: str
    total: int
    completed: int
    failed: int
    created_at: str | None = None
    completed_at: str | None = None


class ExplainabilityAPI:
    """API for decision explainability operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # ==========================================================================
    # Explanations
    # ==========================================================================

    async def get_explanation(
        self,
        debate_id: str,
        *,
        format: Literal["brief", "detailed", "technical"] | None = None,
        include_factors: bool = True,
    ) -> Explanation:
        """Get explanation for a debate decision.

        Args:
            debate_id: Debate ID
            format: Explanation format (brief, detailed, technical)
            include_factors: Whether to include contributing factors

        Returns:
            Decision explanation
        """
        params: dict[str, Any] = {"include_factors": include_factors}
        if format:
            params["format"] = format

        data = await self._client._get(
            f"/api/v1/debates/{debate_id}/explanation", params=params
        )
        return Explanation.model_validate(data)

    async def get_factors(
        self,
        debate_id: str,
        *,
        min_contribution: float | None = None,
    ) -> list[ExplanationFactor]:
        """Get factors contributing to a debate decision.

        Args:
            debate_id: Debate ID
            min_contribution: Minimum contribution threshold (0-1)

        Returns:
            List of contributing factors
        """
        params: dict[str, Any] = {}
        if min_contribution is not None:
            params["min_contribution"] = min_contribution

        data = await self._client._get(
            f"/api/v1/debates/{debate_id}/explanation/factors",
            params=params or None,
        )
        return [ExplanationFactor.model_validate(f) for f in data.get("factors", [])]

    # ==========================================================================
    # Counterfactuals
    # ==========================================================================

    async def get_counterfactuals(
        self,
        debate_id: str,
        *,
        max_scenarios: int | None = None,
    ) -> list[Counterfactual]:
        """Get counterfactual scenarios for a debate.

        Args:
            debate_id: Debate ID
            max_scenarios: Maximum number of scenarios to generate

        Returns:
            List of counterfactual scenarios
        """
        params: dict[str, Any] = {}
        if max_scenarios is not None:
            params["max_scenarios"] = max_scenarios

        data = await self._client._get(
            f"/api/v1/debates/{debate_id}/counterfactuals",
            params=params or None,
        )
        return [
            Counterfactual.model_validate(c) for c in data.get("counterfactuals", [])
        ]

    async def generate_counterfactual(
        self,
        debate_id: str,
        *,
        variable: str,
        new_value: str,
        context: str | None = None,
    ) -> Counterfactual:
        """Generate a specific counterfactual scenario.

        Args:
            debate_id: Debate ID
            variable: Variable to change
            new_value: New value for the variable
            context: Additional context for generation

        Returns:
            Generated counterfactual scenario
        """
        body: dict[str, Any] = {
            "variable": variable,
            "new_value": new_value,
        }
        if context:
            body["context"] = context

        data = await self._client._post(
            f"/api/v1/debates/{debate_id}/counterfactuals/generate", body
        )
        return Counterfactual.model_validate(data)

    # ==========================================================================
    # Provenance
    # ==========================================================================

    async def get_provenance(
        self,
        debate_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ProvenanceEntry]:
        """Get provenance chain for a debate.

        Args:
            debate_id: Debate ID
            limit: Maximum number of entries
            offset: Pagination offset

        Returns:
            List of provenance entries
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/debates/{debate_id}/provenance", params=params
        )
        return [ProvenanceEntry.model_validate(p) for p in data.get("entries", [])]

    # ==========================================================================
    # Narratives
    # ==========================================================================

    async def get_narrative(
        self,
        debate_id: str,
        *,
        format: Literal["brief", "detailed", "executive_summary"] = "detailed",
    ) -> Narrative:
        """Get narrative explanation for a debate.

        Args:
            debate_id: Debate ID
            format: Narrative format

        Returns:
            Decision narrative
        """
        params: dict[str, Any] = {"format": format}
        data = await self._client._get(
            f"/api/v1/debates/{debate_id}/narrative", params=params
        )
        return Narrative.model_validate(data)

    # ==========================================================================
    # Batch Operations
    # ==========================================================================

    async def create_batch_explanation(
        self,
        debate_ids: list[str],
        *,
        format: Literal["brief", "detailed", "technical"] = "detailed",
        include_factors: bool = True,
        include_counterfactuals: bool = False,
    ) -> BatchExplanationJob:
        """Create batch explanation job for multiple debates.

        Args:
            debate_ids: List of debate IDs
            format: Explanation format
            include_factors: Include contributing factors
            include_counterfactuals: Include counterfactual scenarios

        Returns:
            Batch job status
        """
        body: dict[str, Any] = {
            "debate_ids": debate_ids,
            "format": format,
            "include_factors": include_factors,
            "include_counterfactuals": include_counterfactuals,
        }
        data = await self._client._post("/api/v1/explanations/batch", body)
        return BatchExplanationJob.model_validate(data)

    async def get_batch_status(self, batch_id: str) -> BatchExplanationJob:
        """Get status of a batch explanation job.

        Args:
            batch_id: Batch job ID

        Returns:
            Batch job status
        """
        data = await self._client._get(f"/api/v1/explanations/batch/{batch_id}")
        return BatchExplanationJob.model_validate(data)

    async def get_batch_results(
        self,
        batch_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Explanation]:
        """Get results from a batch explanation job.

        Args:
            batch_id: Batch job ID
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of explanations
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/explanations/batch/{batch_id}/results", params=params
        )
        return [Explanation.model_validate(e) for e in data.get("results", [])]

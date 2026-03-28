"""Matrix debates API resource for parallel scenario debates."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.client.client import AragoraClient

from aragora.client.models import (
    MatrixConclusion,
    MatrixDebate,
    MatrixDebateCreateRequest,
    MatrixDebateCreateResponse,
    MatrixModelCombination,
    MatrixScenario,
)


class MatrixDebatesAPI:
    """API interface for matrix debates with parallel scenarios."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def create(
        self,
        task: str,
        agents: list[str] | None = None,
        scenarios: list[dict[str, Any]] | None = None,
        model_combinations: list[dict[str, Any] | list[str]] | None = None,
        max_rounds: int = 3,
    ) -> MatrixDebateCreateResponse:
        """
        Create and start a matrix debate with parallel scenarios.

        Matrix debates run the same debate across different scenarios
        to identify universal vs conditional conclusions.

        Args:
            task: The base question or topic to debate.
            agents: List of agent IDs to participate.
            scenarios: List of scenario configurations.
                Each scenario can have: name, parameters, constraints, agents, is_baseline.
            model_combinations: Optional agent combinations to sweep across.
                Each entry can be either a list of agent IDs or an object with
                name, agents, and is_baseline.
            max_rounds: Maximum rounds per scenario (1-10).

        Returns:
            MatrixDebateCreateResponse with matrix_id.
        """
        scenario_models = []
        if scenarios:
            for s in scenarios:
                scenario_models.append(MatrixScenario(**s))
        combination_models = []
        if model_combinations:
            for combo in model_combinations:
                if isinstance(combo, list):
                    combination_models.append(MatrixModelCombination(agents=combo))
                else:
                    combination_models.append(MatrixModelCombination(**combo))

        request = MatrixDebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            scenarios=scenario_models,
            model_combinations=combination_models,
            max_rounds=max_rounds,
        )

        response = self._client._post("/api/v1/debates/matrix", request.model_dump())
        return MatrixDebateCreateResponse(**response)

    async def create_async(
        self,
        task: str,
        agents: list[str] | None = None,
        scenarios: list[dict[str, Any]] | None = None,
        model_combinations: list[dict[str, Any] | list[str]] | None = None,
        max_rounds: int = 3,
    ) -> MatrixDebateCreateResponse:
        """Async version of create()."""
        scenario_models = []
        if scenarios:
            for s in scenarios:
                scenario_models.append(MatrixScenario(**s))
        combination_models = []
        if model_combinations:
            for combo in model_combinations:
                if isinstance(combo, list):
                    combination_models.append(MatrixModelCombination(agents=combo))
                else:
                    combination_models.append(MatrixModelCombination(**combo))

        request = MatrixDebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            scenarios=scenario_models,
            model_combinations=combination_models,
            max_rounds=max_rounds,
        )

        response = await self._client._post_async("/api/v1/debates/matrix", request.model_dump())
        return MatrixDebateCreateResponse(**response)

    def get(self, matrix_id: str) -> MatrixDebate:
        """
        Get matrix debate details by ID.

        Args:
            matrix_id: The matrix debate ID.

        Returns:
            MatrixDebate with full details including scenario results.
        """
        response = self._client._get(f"/api/v1/debates/matrix/{matrix_id}")
        return MatrixDebate(**response)

    async def get_async(self, matrix_id: str) -> MatrixDebate:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/v1/debates/matrix/{matrix_id}")
        return MatrixDebate(**response)

    def get_conclusions(self, matrix_id: str) -> MatrixConclusion:
        """
        Get universal and conditional conclusions from a matrix debate.

        Args:
            matrix_id: The matrix debate ID.

        Returns:
            MatrixConclusion with universal, conditional, and contradictory findings.
        """
        response = self._client._get(f"/api/v1/debates/matrix/{matrix_id}/conclusions")
        return MatrixConclusion(**response)

    async def get_conclusions_async(self, matrix_id: str) -> MatrixConclusion:
        """Async version of get_conclusions()."""
        response = await self._client._get_async(f"/api/v1/debates/matrix/{matrix_id}/conclusions")
        return MatrixConclusion(**response)


__all__ = ["MatrixDebatesAPI"]

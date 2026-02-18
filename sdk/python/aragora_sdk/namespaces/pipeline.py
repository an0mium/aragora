"""Pipeline namespace API (Idea-to-Execution endpoints)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PipelineAPI:
    """Synchronous Pipeline API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def run(
        self,
        input_text: str,
        *,
        stages: list[str] | None = None,
        debate_rounds: int = 3,
        workflow_mode: str = "quick",
        dry_run: bool = False,
        enable_receipts: bool = True,
    ) -> dict[str, Any]:
        """Start an async pipeline execution.

        Args:
            input_text: The idea/problem statement to process
            stages: Stages to run (default: all 4)
            debate_rounds: Number of debate rounds for ideation
            workflow_mode: "quick" or "debate"
            dry_run: If True, skip orchestration
            enable_receipts: Generate DecisionReceipt on completion

        Returns:
            Pipeline ID and initial status
        """
        payload: dict[str, Any] = {
            "input_text": input_text,
            "debate_rounds": debate_rounds,
            "workflow_mode": workflow_mode,
            "dry_run": dry_run,
            "enable_receipts": enable_receipts,
        }
        if stages:
            payload["stages"] = stages
        return self._client.request("POST", "/api/v1/canvas/pipeline/run", json=payload)

    def from_debate(
        self,
        cartographer_data: dict[str, Any],
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run full pipeline from ArgumentCartographer debate export."""
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-debate",
            json={"cartographer_data": cartographer_data, "auto_advance": auto_advance},
        )

    def from_ideas(
        self,
        ideas: list[str],
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run full pipeline from raw idea strings."""
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-ideas",
            json={"ideas": ideas, "auto_advance": auto_advance},
        )

    def status(self, pipeline_id: str) -> dict[str, Any]:
        """Get pipeline per-stage status."""
        return self._client.request("GET", f"/api/v1/canvas/pipeline/{pipeline_id}/status")

    def get(self, pipeline_id: str) -> dict[str, Any]:
        """Get full pipeline result."""
        return self._client.request("GET", f"/api/v1/canvas/pipeline/{pipeline_id}")

    def graph(self, pipeline_id: str, *, stage: str | None = None) -> dict[str, Any]:
        """Get React Flow JSON graph for pipeline stages."""
        params = {"stage": stage} if stage else {}
        return self._client.request(
            "GET", f"/api/v1/canvas/pipeline/{pipeline_id}/graph", params=params,
        )

    def receipt(self, pipeline_id: str) -> dict[str, Any]:
        """Get DecisionReceipt for a completed pipeline."""
        return self._client.request("GET", f"/api/v1/canvas/pipeline/{pipeline_id}/receipt")

    def advance(self, pipeline_id: str, target_stage: str) -> dict[str, Any]:
        """Advance a pipeline to the next stage."""
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/advance",
            json={"pipeline_id": pipeline_id, "target_stage": target_stage},
        )


class AsyncPipelineAPI:
    """Asynchronous Pipeline API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def run(
        self,
        input_text: str,
        *,
        stages: list[str] | None = None,
        debate_rounds: int = 3,
        workflow_mode: str = "quick",
        dry_run: bool = False,
        enable_receipts: bool = True,
    ) -> dict[str, Any]:
        """Start an async pipeline execution."""
        payload: dict[str, Any] = {
            "input_text": input_text,
            "debate_rounds": debate_rounds,
            "workflow_mode": workflow_mode,
            "dry_run": dry_run,
            "enable_receipts": enable_receipts,
        }
        if stages:
            payload["stages"] = stages
        return await self._client.request("POST", "/api/v1/canvas/pipeline/run", json=payload)

    async def from_debate(
        self,
        cartographer_data: dict[str, Any],
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run full pipeline from ArgumentCartographer debate export."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-debate",
            json={"cartographer_data": cartographer_data, "auto_advance": auto_advance},
        )

    async def from_ideas(
        self,
        ideas: list[str],
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run full pipeline from raw idea strings."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-ideas",
            json={"ideas": ideas, "auto_advance": auto_advance},
        )

    async def status(self, pipeline_id: str) -> dict[str, Any]:
        """Get pipeline per-stage status."""
        return await self._client.request("GET", f"/api/v1/canvas/pipeline/{pipeline_id}/status")

    async def get(self, pipeline_id: str) -> dict[str, Any]:
        """Get full pipeline result."""
        return await self._client.request("GET", f"/api/v1/canvas/pipeline/{pipeline_id}")

    async def graph(self, pipeline_id: str, *, stage: str | None = None) -> dict[str, Any]:
        """Get React Flow JSON graph for pipeline stages."""
        params = {"stage": stage} if stage else {}
        return await self._client.request(
            "GET", f"/api/v1/canvas/pipeline/{pipeline_id}/graph", params=params,
        )

    async def receipt(self, pipeline_id: str) -> dict[str, Any]:
        """Get DecisionReceipt for a completed pipeline."""
        return await self._client.request("GET", f"/api/v1/canvas/pipeline/{pipeline_id}/receipt")

    async def advance(self, pipeline_id: str, target_stage: str) -> dict[str, Any]:
        """Advance a pipeline to the next stage."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/advance",
            json={"pipeline_id": pipeline_id, "target_stage": target_stage},
        )

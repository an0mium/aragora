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
        use_ai: bool = False,
    ) -> dict[str, Any]:
        """Start an async pipeline execution.

        Args:
            input_text: The idea/problem statement to process
            stages: Stages to run (default: all 4)
            debate_rounds: Number of debate rounds for ideation
            workflow_mode: "quick" or "debate"
            dry_run: If True, skip orchestration
            enable_receipts: Generate DecisionReceipt on completion
            use_ai: If True, use AI-assisted goal extraction

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
        if use_ai:
            payload["use_ai"] = True
        if stages:
            payload["stages"] = stages
        return self._client.request("POST", "/api/v1/canvas/pipeline/run", json=payload)

    def from_debate(
        self,
        cartographer_data: dict[str, Any],
        auto_advance: bool = True,
        use_ai: bool = False,
    ) -> dict[str, Any]:
        """Run full pipeline from ArgumentCartographer debate export."""
        payload: dict[str, Any] = {
            "cartographer_data": cartographer_data,
            "auto_advance": auto_advance,
        }
        if use_ai:
            payload["use_ai"] = True
        return self._client.request(
            "POST", "/api/v1/canvas/pipeline/from-debate", json=payload,
        )

    def from_ideas(
        self,
        ideas: list[str],
        auto_advance: bool = True,
        use_ai: bool = False,
    ) -> dict[str, Any]:
        """Run full pipeline from raw idea strings."""
        payload: dict[str, Any] = {
            "ideas": ideas,
            "auto_advance": auto_advance,
        }
        if use_ai:
            payload["use_ai"] = True
        return self._client.request(
            "POST", "/api/v1/canvas/pipeline/from-ideas", json=payload,
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

    def stage(self, pipeline_id: str, stage: str) -> dict[str, Any]:
        """Get a specific stage canvas from a pipeline."""
        return self._client.request(
            "GET", f"/api/v1/canvas/pipeline/{pipeline_id}/stage/{stage}",
        )

    def extract_goals(
        self,
        ideas_canvas_id: str,
        *,
        ideas_canvas_data: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract goals from an ideas canvas."""
        payload: dict[str, Any] = {"ideas_canvas_id": ideas_canvas_id}
        if ideas_canvas_data:
            payload["ideas_canvas_data"] = ideas_canvas_data
        if config:
            payload["config"] = config
        return self._client.request(
            "POST", "/api/v1/canvas/pipeline/extract-goals", json=payload,
        )

    def approve_transition(
        self, pipeline_id: str, *, approved: bool = True, notes: str | None = None,
    ) -> dict[str, Any]:
        """Approve or reject a pending stage transition."""
        payload: dict[str, Any] = {"approved": approved}
        if notes:
            payload["notes"] = notes
        return self._client.request(
            "POST",
            f"/api/v1/canvas/pipeline/{pipeline_id}/approve-transition",
            json=payload,
        )

    def save(self, pipeline_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Save/update a pipeline."""
        return self._client.request(
            "PUT", f"/api/v1/canvas/pipeline/{pipeline_id}", json=data,
        )

    def convert_debate(self, cartographer_data: dict[str, Any]) -> dict[str, Any]:
        """Convert ArgumentCartographer debate to React Flow ideas canvas."""
        return self._client.request(
            "POST",
            "/api/v1/canvas/convert/debate",
            json={"cartographer_data": cartographer_data},
        )

    def convert_workflow(self, workflow_data: dict[str, Any]) -> dict[str, Any]:
        """Convert WorkflowDefinition to React Flow actions canvas."""
        return self._client.request(
            "POST",
            "/api/v1/canvas/convert/workflow",
            json={"workflow_data": workflow_data},
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
        use_ai: bool = False,
    ) -> dict[str, Any]:
        """Start an async pipeline execution."""
        payload: dict[str, Any] = {
            "input_text": input_text,
            "debate_rounds": debate_rounds,
            "workflow_mode": workflow_mode,
            "dry_run": dry_run,
            "enable_receipts": enable_receipts,
        }
        if use_ai:
            payload["use_ai"] = True
        if stages:
            payload["stages"] = stages
        return await self._client.request("POST", "/api/v1/canvas/pipeline/run", json=payload)

    async def from_debate(
        self,
        cartographer_data: dict[str, Any],
        auto_advance: bool = True,
        use_ai: bool = False,
    ) -> dict[str, Any]:
        """Run full pipeline from ArgumentCartographer debate export."""
        payload: dict[str, Any] = {
            "cartographer_data": cartographer_data,
            "auto_advance": auto_advance,
        }
        if use_ai:
            payload["use_ai"] = True
        return await self._client.request(
            "POST", "/api/v1/canvas/pipeline/from-debate", json=payload,
        )

    async def from_ideas(
        self,
        ideas: list[str],
        auto_advance: bool = True,
        use_ai: bool = False,
    ) -> dict[str, Any]:
        """Run full pipeline from raw idea strings."""
        payload: dict[str, Any] = {
            "ideas": ideas,
            "auto_advance": auto_advance,
        }
        if use_ai:
            payload["use_ai"] = True
        return await self._client.request(
            "POST", "/api/v1/canvas/pipeline/from-ideas", json=payload,
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

    async def stage(self, pipeline_id: str, stage: str) -> dict[str, Any]:
        """Get a specific stage canvas from a pipeline."""
        return await self._client.request(
            "GET", f"/api/v1/canvas/pipeline/{pipeline_id}/stage/{stage}",
        )

    async def extract_goals(
        self,
        ideas_canvas_id: str,
        *,
        ideas_canvas_data: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract goals from an ideas canvas."""
        payload: dict[str, Any] = {"ideas_canvas_id": ideas_canvas_id}
        if ideas_canvas_data:
            payload["ideas_canvas_data"] = ideas_canvas_data
        if config:
            payload["config"] = config
        return await self._client.request(
            "POST", "/api/v1/canvas/pipeline/extract-goals", json=payload,
        )

    async def approve_transition(
        self, pipeline_id: str, *, approved: bool = True, notes: str | None = None,
    ) -> dict[str, Any]:
        """Approve or reject a pending stage transition."""
        payload: dict[str, Any] = {"approved": approved}
        if notes:
            payload["notes"] = notes
        return await self._client.request(
            "POST",
            f"/api/v1/canvas/pipeline/{pipeline_id}/approve-transition",
            json=payload,
        )

    async def save(self, pipeline_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Save/update a pipeline."""
        return await self._client.request(
            "PUT", f"/api/v1/canvas/pipeline/{pipeline_id}", json=data,
        )

    async def convert_debate(self, cartographer_data: dict[str, Any]) -> dict[str, Any]:
        """Convert ArgumentCartographer debate to React Flow ideas canvas."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/convert/debate",
            json={"cartographer_data": cartographer_data},
        )

    async def convert_workflow(self, workflow_data: dict[str, Any]) -> dict[str, Any]:
        """Convert WorkflowDefinition to React Flow actions canvas."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/convert/workflow",
            json={"workflow_data": workflow_data},
        )

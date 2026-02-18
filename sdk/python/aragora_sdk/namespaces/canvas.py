"""
Canvas Namespace API

Provides methods for the idea-to-execution canvas pipeline:
- Run pipelines from debate results or raw ideas
- Advance stages with human-in-the-loop control
- Retrieve pipeline results and individual stage canvases
- Convert debate/workflow data to React Flow canvas format
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

PipelineStage = Literal["ideas", "goals", "actions", "orchestration"]


class CanvasAPI:
    """Synchronous Canvas Pipeline API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def run_from_debate(
        self,
        cartographer_data: dict[str, Any],
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run full pipeline from an ArgumentCartographer debate export.

        Transforms debate argument graphs into actionable execution plans
        through 4 stages: ideas -> goals -> actions -> orchestration.

        Args:
            cartographer_data: Debate graph from ArgumentCartographer.export()
            auto_advance: If True, advance through all stages automatically

        Returns:
            PipelineResult with canvases for each completed stage
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-debate",
            json={
                "cartographer_data": cartographer_data,
                "auto_advance": auto_advance,
            },
        )

    def run_from_ideas(
        self,
        ideas: list[str],
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run full pipeline from raw idea strings.

        Simpler entry point that skips debate graph parsing.

        Args:
            ideas: List of idea strings to process
            auto_advance: If True, advance through all stages automatically

        Returns:
            PipelineResult with canvases for each completed stage
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-ideas",
            json={
                "ideas": ideas,
                "auto_advance": auto_advance,
            },
        )

    def advance_stage(
        self,
        pipeline_id: str,
        target_stage: PipelineStage,
    ) -> dict[str, Any]:
        """Advance pipeline to the next stage after human review.

        Used for human-in-the-loop workflows where each stage
        requires approval before proceeding.

        Args:
            pipeline_id: Pipeline identifier from a previous run
            target_stage: Stage to advance to

        Returns:
            Updated PipelineResult
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/advance",
            json={
                "pipeline_id": pipeline_id,
                "target_stage": target_stage,
            },
        )

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        """Get complete pipeline result by ID.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Full PipelineResult with all stage canvases
        """
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}",
        )

    def get_stage(
        self,
        pipeline_id: str,
        stage: PipelineStage,
    ) -> dict[str, Any]:
        """Get a specific stage canvas from a pipeline.

        Args:
            pipeline_id: Pipeline identifier
            stage: Stage to retrieve (ideas, goals, actions, orchestration)

        Returns:
            Canvas data in React Flow format (nodes + edges)
        """
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/stage/{stage}",
        )

    def convert_debate(
        self,
        cartographer_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert debate argument graph to ideas canvas.

        Standalone conversion without running the full pipeline.

        Args:
            cartographer_data: Debate graph from ArgumentCartographer

        Returns:
            Ideas canvas in React Flow format
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/convert/debate",
            json={"cartographer_data": cartographer_data},
        )

    def convert_workflow(
        self,
        workflow_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert workflow definition to actions canvas.

        Standalone conversion without running the full pipeline.

        Args:
            workflow_data: Workflow definition to convert

        Returns:
            Actions canvas in React Flow format
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/convert/workflow",
            json={"workflow_data": workflow_data},
        )


class AsyncCanvasAPI:
    """Asynchronous Canvas Pipeline API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def run_from_debate(
        self,
        cartographer_data: dict[str, Any],
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run full pipeline from an ArgumentCartographer debate export."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-debate",
            json={
                "cartographer_data": cartographer_data,
                "auto_advance": auto_advance,
            },
        )

    async def run_from_ideas(
        self,
        ideas: list[str],
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run full pipeline from raw idea strings."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-ideas",
            json={
                "ideas": ideas,
                "auto_advance": auto_advance,
            },
        )

    async def advance_stage(
        self,
        pipeline_id: str,
        target_stage: PipelineStage,
    ) -> dict[str, Any]:
        """Advance pipeline to the next stage after human review."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/advance",
            json={
                "pipeline_id": pipeline_id,
                "target_stage": target_stage,
            },
        )

    async def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        """Get complete pipeline result by ID."""
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}",
        )

    async def get_stage(
        self,
        pipeline_id: str,
        stage: PipelineStage,
    ) -> dict[str, Any]:
        """Get a specific stage canvas from a pipeline."""
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/stage/{stage}",
        )

    async def convert_debate(
        self,
        cartographer_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert debate argument graph to ideas canvas."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/convert/debate",
            json={"cartographer_data": cartographer_data},
        )

    async def convert_workflow(
        self,
        workflow_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert workflow definition to actions canvas."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/convert/workflow",
            json={"workflow_data": workflow_data},
        )

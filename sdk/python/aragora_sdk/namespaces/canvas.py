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

    def run_from_braindump(
        self,
        text: str,
        *,
        context: str | None = None,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run pipeline from raw unstructured text.

        Parses text into ideas, then runs through the full pipeline.

        Args:
            text: Raw text to parse into ideas
            context: Optional context for idea extraction
            auto_advance: If True, advance through all stages

        Returns:
            PipelineResult with parsed ideas and stage canvases
        """
        payload: dict[str, Any] = {
            "text": text,
            "auto_advance": auto_advance,
        }
        if context:
            payload["context"] = context
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-braindump",
            json=payload,
        )

    def run_from_template(
        self,
        template_name: str,
        *,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run pipeline from a named template.

        Args:
            template_name: Name of the pipeline template
            auto_advance: If True, advance through all stages

        Returns:
            PipelineResult from template execution
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-template",
            json={
                "template_name": template_name,
                "auto_advance": auto_advance,
            },
        )

    def execute_pipeline(
        self,
        pipeline_id: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute a pipeline's orchestration stage.

        Args:
            pipeline_id: Pipeline identifier
            dry_run: If True, return execution plan without running

        Returns:
            Execution status with agent task assignments
        """
        return self._client.request(
            "POST",
            f"/api/v1/canvas/pipeline/{pipeline_id}/execute",
            json={"dry_run": dry_run},
        )

    def list_templates(self, *, category: str | None = None) -> dict[str, Any]:
        """List available pipeline templates.

        Args:
            category: Optional category filter

        Returns:
            Template list with count
        """
        params = {"category": category} if category else {}
        return self._client.request(
            "GET",
            "/api/v1/canvas/pipeline/templates",
            params=params,
        )

    def get_receipt(self, pipeline_id: str) -> dict[str, Any]:
        """Get DecisionReceipt for a completed pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Decision receipt with audit trail
        """
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/receipt",
        )

    def get_graph(
        self,
        pipeline_id: str,
        *,
        stage: str | None = None,
    ) -> dict[str, Any]:
        """Get React Flow graph for pipeline stages.

        Args:
            pipeline_id: Pipeline identifier
            stage: Optional specific stage to retrieve

        Returns:
            Graph data with nodes and edges
        """
        params = {"stage": stage} if stage else {}
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/graph",
            params=params,
        )

    def get_status(self, pipeline_id: str) -> dict[str, Any]:
        """Get pipeline per-stage status.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Status for each stage including duration
        """
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/status",
        )

    def debate_to_pipeline(
        self,
        debate_id: str,
        *,
        use_universal: bool = False,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Convert an existing debate into a pipeline.

        Args:
            debate_id: ID of the debate to convert
            use_universal: If True, build universal execution graph
            auto_advance: If True, advance through all stages

        Returns:
            PipelineResult with source debate reference
        """
        return self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/to-pipeline",
            json={
                "use_universal": use_universal,
                "auto_advance": auto_advance,
            },
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

    async def run_from_braindump(
        self,
        text: str,
        *,
        context: str | None = None,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run pipeline from raw unstructured text."""
        payload: dict[str, Any] = {
            "text": text,
            "auto_advance": auto_advance,
        }
        if context:
            payload["context"] = context
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-braindump",
            json=payload,
        )

    async def run_from_template(
        self,
        template_name: str,
        *,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run pipeline from a named template."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-template",
            json={
                "template_name": template_name,
                "auto_advance": auto_advance,
            },
        )

    async def execute_pipeline(
        self,
        pipeline_id: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute a pipeline's orchestration stage."""
        return await self._client.request(
            "POST",
            f"/api/v1/canvas/pipeline/{pipeline_id}/execute",
            json={"dry_run": dry_run},
        )

    async def list_templates(self, *, category: str | None = None) -> dict[str, Any]:
        """List available pipeline templates."""
        params = {"category": category} if category else {}
        return await self._client.request(
            "GET",
            "/api/v1/canvas/pipeline/templates",
            params=params,
        )

    async def get_receipt(self, pipeline_id: str) -> dict[str, Any]:
        """Get DecisionReceipt for a completed pipeline."""
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/receipt",
        )

    async def get_graph(
        self,
        pipeline_id: str,
        *,
        stage: str | None = None,
    ) -> dict[str, Any]:
        """Get React Flow graph for pipeline stages."""
        params = {"stage": stage} if stage else {}
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/graph",
            params=params,
        )

    async def get_status(self, pipeline_id: str) -> dict[str, Any]:
        """Get pipeline per-stage status."""
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/status",
        )

    async def debate_to_pipeline(
        self,
        debate_id: str,
        *,
        use_universal: bool = False,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Convert an existing debate into a pipeline."""
        return await self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/to-pipeline",
            json={
                "use_universal": use_universal,
                "auto_advance": auto_advance,
            },
        )

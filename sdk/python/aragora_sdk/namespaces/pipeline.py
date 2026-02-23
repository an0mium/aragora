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
        self,
        pipeline_id: str,
        from_stage: str,
        to_stage: str,
        *,
        approved: bool = True,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Approve or reject a pending stage transition.

        Args:
            pipeline_id: Pipeline identifier
            from_stage: Source stage of the transition
            to_stage: Target stage of the transition
            approved: If True, approve; if False, reject
            comment: Optional comment explaining the decision
        """
        payload: dict[str, Any] = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "approved": approved,
        }
        if comment:
            payload["comment"] = comment
        return self._client.request(
            "POST",
            f"/api/v1/canvas/pipeline/{pipeline_id}/approve-transition",
            json=payload,
        )

    def from_braindump(
        self,
        text: str,
        *,
        context: str | None = None,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run pipeline from a raw text braindump.

        Parses unstructured text into ideas, then processes through
        the full pipeline.

        Args:
            text: Raw text to parse into ideas
            context: Optional context for idea extraction
            auto_advance: If True, advance through all stages
        """
        payload: dict[str, Any] = {
            "text": text,
            "auto_advance": auto_advance,
        }
        if context:
            payload["context"] = context
        return self._client.request(
            "POST", "/api/v1/canvas/pipeline/from-braindump", json=payload,
        )

    def from_template(
        self,
        template_name: str,
        *,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run pipeline from a named template.

        Args:
            template_name: Name of the pipeline template
            auto_advance: If True, advance through all stages
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-template",
            json={
                "template_name": template_name,
                "auto_advance": auto_advance,
            },
        )

    def execute(
        self,
        pipeline_id: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute a pipeline's orchestration stage.

        Args:
            pipeline_id: Pipeline identifier
            dry_run: If True, return execution plan without running
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
        """
        params = {"category": category} if category else {}
        return self._client.request(
            "GET", "/api/v1/canvas/pipeline/templates", params=params,
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
        """
        return self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/to-pipeline",
            json={
                "use_universal": use_universal,
                "auto_advance": auto_advance,
            },
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

    # =========================================================================
    # Pipeline Graphs & Transitions
    # =========================================================================

    def get_graph(self) -> dict[str, Any]:
        """Get the current pipeline execution graph."""
        return self._client.request("GET", "/api/v1/pipeline/graph")

    def list_graphs(self) -> dict[str, Any]:
        """List saved pipeline graphs."""
        return self._client.request("GET", "/api/v1/pipeline/graphs")

    def create_graph(self, graph_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new pipeline graph.

        Args:
            graph_data: Graph definition (nodes, edges, metadata).

        Returns:
            Created graph with ID.
        """
        return self._client.request("POST", "/api/v1/pipeline/graphs", json=graph_data)

    def update_graph(self, graph_data: dict[str, Any]) -> dict[str, Any]:
        """Update an existing pipeline graph.

        Args:
            graph_data: Updated graph definition (must include graph_id).

        Returns:
            Updated graph.
        """
        return self._client.request("PUT", "/api/v1/pipeline/graphs", json=graph_data)

    def delete_graph(self, **kwargs: Any) -> dict[str, Any]:
        """Delete a pipeline graph.

        Args:
            **kwargs: Delete parameters (graph_id, etc.).

        Returns:
            Deletion confirmation.
        """
        return self._client.request("DELETE", "/api/v1/pipeline/graphs", json=kwargs)

    def create_transition(
        self,
        from_stage: str,
        to_stage: str,
        *,
        pipeline_id: str | None = None,
        conditions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a pipeline stage transition.

        Args:
            from_stage: Source stage.
            to_stage: Target stage.
            pipeline_id: Pipeline identifier (optional).
            conditions: Transition conditions/guards.

        Returns:
            Created transition details.
        """
        payload: dict[str, Any] = {
            "from_stage": from_stage,
            "to_stage": to_stage,
        }
        if pipeline_id:
            payload["pipeline_id"] = pipeline_id
        if conditions:
            payload["conditions"] = conditions
        return self._client.request("POST", "/api/v1/pipeline/transitions", json=payload)


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
        self,
        pipeline_id: str,
        from_stage: str,
        to_stage: str,
        *,
        approved: bool = True,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Approve or reject a pending stage transition."""
        payload: dict[str, Any] = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "approved": approved,
        }
        if comment:
            payload["comment"] = comment
        return await self._client.request(
            "POST",
            f"/api/v1/canvas/pipeline/{pipeline_id}/approve-transition",
            json=payload,
        )

    async def from_braindump(
        self,
        text: str,
        *,
        context: str | None = None,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Run pipeline from a raw text braindump."""
        payload: dict[str, Any] = {
            "text": text,
            "auto_advance": auto_advance,
        }
        if context:
            payload["context"] = context
        return await self._client.request(
            "POST", "/api/v1/canvas/pipeline/from-braindump", json=payload,
        )

    async def from_template(
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

    async def execute(
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
            "GET", "/api/v1/canvas/pipeline/templates", params=params,
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

    # =========================================================================
    # Pipeline Graphs & Transitions
    # =========================================================================

    async def get_graph(self) -> dict[str, Any]:
        """Get the current pipeline execution graph."""
        return await self._client.request("GET", "/api/v1/pipeline/graph")

    async def list_graphs(self) -> dict[str, Any]:
        """List saved pipeline graphs."""
        return await self._client.request("GET", "/api/v1/pipeline/graphs")

    async def create_graph(self, graph_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new pipeline graph."""
        return await self._client.request("POST", "/api/v1/pipeline/graphs", json=graph_data)

    async def update_graph(self, graph_data: dict[str, Any]) -> dict[str, Any]:
        """Update an existing pipeline graph."""
        return await self._client.request("PUT", "/api/v1/pipeline/graphs", json=graph_data)

    async def delete_graph(self, **kwargs: Any) -> dict[str, Any]:
        """Delete a pipeline graph."""
        return await self._client.request("DELETE", "/api/v1/pipeline/graphs", json=kwargs)

    async def create_transition(
        self,
        from_stage: str,
        to_stage: str,
        *,
        pipeline_id: str | None = None,
        conditions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a pipeline stage transition."""
        payload: dict[str, Any] = {
            "from_stage": from_stage,
            "to_stage": to_stage,
        }
        if pipeline_id:
            payload["pipeline_id"] = pipeline_id
        if conditions:
            payload["conditions"] = conditions
        return await self._client.request("POST", "/api/v1/pipeline/transitions", json=payload)

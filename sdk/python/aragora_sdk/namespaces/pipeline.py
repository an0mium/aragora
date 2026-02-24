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
            "POST",
            "/api/v1/canvas/pipeline/from-debate",
            json=payload,
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
            "POST",
            "/api/v1/canvas/pipeline/from-ideas",
            json=payload,
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
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/graph",
            params=params,
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
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/stage/{stage}",
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
            "POST",
            "/api/v1/canvas/pipeline/extract-goals",
            json=payload,
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
            "POST",
            "/api/v1/canvas/pipeline/from-braindump",
            json=payload,
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
            "GET",
            "/api/v1/canvas/pipeline/templates",
            params=params,
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
            "PUT",
            f"/api/v1/canvas/pipeline/{pipeline_id}",
            json=data,
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
    # Demo & Auto-Run
    # =========================================================================

    def demo(
        self,
        *,
        ideas: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a pre-populated demo pipeline with all 4 stages complete.

        No API keys or authentication required. The pipeline is stored
        server-side so that subsequent execute calls work.

        Args:
            ideas: Custom demo ideas (optional, defaults to built-in examples).

        Returns:
            Demo pipeline result with pipeline_id.
        """
        payload: dict[str, Any] = {}
        if ideas:
            payload["ideas"] = ideas
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/demo",
            json=payload,
        )

    def auto_run(
        self,
        text: str,
        *,
        automation_level: str = "full",
    ) -> dict[str, Any]:
        """Accept unstructured text and auto-run the full pipeline.

        Returns pipeline_id immediately while processing streams via WebSocket.

        Args:
            text: Unstructured text input.
            automation_level: Level of automation (full, guided, manual).

        Returns:
            Pipeline ID and initial status.
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/auto-run",
            json={"text": text, "automation_level": automation_level},
        )

    def extract_principles(
        self,
        ideas_canvas: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract principles/values from an ideas canvas.

        Args:
            ideas_canvas: Ideas canvas data (nodes + edges).

        Returns:
            Extracted principles list.
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/extract-principles",
            json={"ideas_canvas": ideas_canvas},
        )

    def from_system_metrics(self) -> dict[str, Any]:
        """Auto-generate pipeline from system health analysis.

        Returns:
            Pipeline result derived from current system metrics.
        """
        return self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-system-metrics",
            json={},
        )

    # =========================================================================
    # Pipeline Intelligence
    # =========================================================================

    def get_intelligence(self, pipeline_id: str) -> dict[str, Any]:
        """Get per-node intelligence: beliefs, crux status, evidence, precedents.

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            Dict with beliefs, explanations, and precedents arrays.
        """
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/intelligence",
        )

    def get_beliefs(self, pipeline_id: str) -> dict[str, Any]:
        """Get belief network state for pipeline nodes.

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            Dict with beliefs array containing confidence and crux status.
        """
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/beliefs",
        )

    def get_explanations(self, pipeline_id: str) -> dict[str, Any]:
        """Get explainability factors for pipeline decisions.

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            Dict with explanations array.
        """
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/explanations",
        )

    def get_precedents(self, pipeline_id: str) -> dict[str, Any]:
        """Get Knowledge Mound precedents for pipeline goals.

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            Dict with precedents array containing node matches.
        """
        return self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/precedents",
        )

    def self_improve(
        self,
        pipeline_id: str,
        *,
        budget_limit: float = 10.0,
        require_approval: bool = True,
    ) -> dict[str, Any]:
        """Feed a completed pipeline into the self-improvement system.

        Triggers autonomous execution with safety rails (worktree isolation,
        gauntlet validation, regression detection).

        Args:
            pipeline_id: Pipeline identifier.
            budget_limit: Max spend in dollars (default 10.0).
            require_approval: Whether human approval is required (default True).

        Returns:
            Self-improvement run ID and status.
        """
        return self._client.request(
            "POST",
            f"/api/v1/canvas/pipeline/{pipeline_id}/self-improve",
            json={
                "budget_limit": budget_limit,
                "require_approval": require_approval,
            },
        )

    # =========================================================================
    # Pipeline Agents
    # =========================================================================

    def get_agents(self, pipeline_id: str) -> dict[str, Any]:
        """Get active agents with status, worktree, and progress.

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            Dict with agents array and pipeline_id.
        """
        return self._client.request("GET", f"/api/v1/pipeline/{pipeline_id}/agents")

    def approve_agent(
        self,
        pipeline_id: str,
        agent_id: str,
        *,
        notes: str = "",
    ) -> dict[str, Any]:
        """Approve an agent's work on a pipeline.

        Args:
            pipeline_id: Pipeline identifier.
            agent_id: Agent identifier.
            notes: Optional approval notes.

        Returns:
            Approval confirmation.
        """
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/{pipeline_id}/agents/{agent_id}/approve",
            json={"notes": notes},
        )

    def reject_agent(
        self,
        pipeline_id: str,
        agent_id: str,
        *,
        feedback: str,
    ) -> dict[str, Any]:
        """Reject an agent's work on a pipeline with feedback.

        Args:
            pipeline_id: Pipeline identifier.
            agent_id: Agent identifier.
            feedback: Required feedback explaining the rejection.

        Returns:
            Rejection confirmation.
        """
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/{pipeline_id}/agents/{agent_id}/reject",
            json={"feedback": feedback},
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

    # =========================================================================
    # Pipeline Transition Helpers
    # =========================================================================

    def ideas_to_goals(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run idea-to-goals transition helper."""
        return self._client.request(
            "POST",
            "/api/v1/pipeline/transitions/ideas-to-goals",
            json=body,
        )

    def goals_to_tasks(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run goals-to-tasks transition helper."""
        return self._client.request(
            "POST",
            "/api/v1/pipeline/transitions/goals-to-tasks",
            json=body,
        )

    def tasks_to_workflow(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run tasks-to-workflow transition helper."""
        return self._client.request(
            "POST",
            "/api/v1/pipeline/transitions/tasks-to-workflow",
            json=body,
        )

    def execute_transitions(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a transition plan."""
        return self._client.request(
            "POST",
            "/api/v1/pipeline/transitions/execute",
            json=body,
        )

    def get_transition_provenance(self, node_id: str) -> dict[str, Any]:
        """Get provenance for a transition node."""
        return self._client.request(
            "GET",
            f"/api/v1/pipeline/transitions/{node_id}/provenance",
        )

    # =========================================================================
    # Universal Pipeline Graph CRUD
    # =========================================================================

    def get_graph_by_id(self, graph_id: str) -> dict[str, Any]:
        """Get a specific universal pipeline graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Graph with nodes, edges, and metadata.
        """
        return self._client.request("GET", f"/api/v1/pipeline/graph/{graph_id}")

    def delete_graph_by_id(self, graph_id: str) -> dict[str, Any]:
        """Delete a universal pipeline graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Deletion confirmation.
        """
        return self._client.request("DELETE", f"/api/v1/pipeline/graph/{graph_id}")

    def add_graph_node(
        self,
        graph_id: str,
        node_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Add a node to a universal pipeline graph.

        Args:
            graph_id: Graph identifier.
            node_data: Node definition with label, type, stage, etc.

        Returns:
            Created node with ID.
        """
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/graph/{graph_id}/node",
            json=node_data,
        )

    def update_graph_node(
        self,
        graph_id: str,
        node_id: str,
        node_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a node in a universal pipeline graph."""
        return self._client.request(
            "PUT",
            f"/api/v1/pipeline/graph/{graph_id}/node/{node_id}",
            json=node_data,
        )

    def delete_graph_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Remove a node from a universal pipeline graph.

        Args:
            graph_id: Graph identifier.
            node_id: Node identifier to remove.

        Returns:
            Deletion confirmation.
        """
        return self._client.request(
            "DELETE",
            f"/api/v1/pipeline/graph/{graph_id}/node/{node_id}",
        )

    def reassign_graph_node(
        self,
        graph_id: str,
        node_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """Reassign an agent on a pipeline graph node.

        Args:
            graph_id: Graph identifier.
            node_id: Node identifier.
            agent_id: New agent to assign.

        Returns:
            Updated node with new agent assignment.
        """
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/graph/{graph_id}/node/{node_id}/reassign",
            json={"agent_id": agent_id},
        )

    def list_graph_nodes(
        self,
        graph_id: str,
        *,
        stage: str | None = None,
        subtype: str | None = None,
    ) -> dict[str, Any]:
        """Query nodes in a pipeline graph with optional filters.

        Args:
            graph_id: Graph identifier.
            stage: Filter by pipeline stage.
            subtype: Filter by node subtype.

        Returns:
            Dict with matching nodes list.
        """
        params: dict[str, Any] = {}
        if stage:
            params["stage"] = stage
        if subtype:
            params["subtype"] = subtype
        return self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/nodes",
            params=params,
        )

    def promote_graph_nodes(
        self,
        graph_id: str,
        node_ids: list[str],
        *,
        target_stage: str | None = None,
    ) -> dict[str, Any]:
        """Promote nodes to the next pipeline stage.

        Args:
            graph_id: Graph identifier.
            node_ids: Node IDs to promote.
            target_stage: Target stage (optional, auto-detected if omitted).

        Returns:
            Promotion result with promoted count and new stage.
        """
        payload: dict[str, Any] = {"node_ids": node_ids}
        if target_stage:
            payload["target_stage"] = target_stage
        return self._client.request(
            "POST",
            f"/api/v1/pipeline/graph/{graph_id}/promote",
            json=payload,
        )

    def get_graph_provenance(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Get the provenance chain for a specific graph node.

        Args:
            graph_id: Graph identifier.
            node_id: Node to trace provenance for.

        Returns:
            Provenance chain with source lineage.
        """
        return self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/provenance/{node_id}",
        )

    def get_graph_react_flow(self, graph_id: str) -> dict[str, Any]:
        """Export a pipeline graph as React Flow JSON.

        Args:
            graph_id: Graph identifier.

        Returns:
            React Flow format with nodes and edges arrays.
        """
        return self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/react-flow",
        )

    def get_graph_integrity(self, graph_id: str) -> dict[str, Any]:
        """Get the integrity hash for a pipeline graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Dict with integrity hash and verification status.
        """
        return self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/integrity",
        )

    def get_graph_suggestions(self, graph_id: str) -> dict[str, Any]:
        """Get transition suggestions for a pipeline graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Dict with suggested next transitions.
        """
        return self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/suggestions",
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
            "POST",
            "/api/v1/canvas/pipeline/from-debate",
            json=payload,
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
            "POST",
            "/api/v1/canvas/pipeline/from-ideas",
            json=payload,
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
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/graph",
            params=params,
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
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/stage/{stage}",
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
            "POST",
            "/api/v1/canvas/pipeline/extract-goals",
            json=payload,
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
            "POST",
            "/api/v1/canvas/pipeline/from-braindump",
            json=payload,
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
            "GET",
            "/api/v1/canvas/pipeline/templates",
            params=params,
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
            "PUT",
            f"/api/v1/canvas/pipeline/{pipeline_id}",
            json=data,
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
    # Demo & Auto-Run
    # =========================================================================

    async def demo(
        self,
        *,
        ideas: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a pre-populated demo pipeline with all 4 stages complete."""
        payload: dict[str, Any] = {}
        if ideas:
            payload["ideas"] = ideas
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/demo",
            json=payload,
        )

    async def auto_run(
        self,
        text: str,
        *,
        automation_level: str = "full",
    ) -> dict[str, Any]:
        """Accept unstructured text and auto-run the full pipeline."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/auto-run",
            json={"text": text, "automation_level": automation_level},
        )

    async def extract_principles(
        self,
        ideas_canvas: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract principles/values from an ideas canvas."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/extract-principles",
            json={"ideas_canvas": ideas_canvas},
        )

    async def from_system_metrics(self) -> dict[str, Any]:
        """Auto-generate pipeline from system health analysis."""
        return await self._client.request(
            "POST",
            "/api/v1/canvas/pipeline/from-system-metrics",
            json={},
        )

    # =========================================================================
    # Pipeline Intelligence
    # =========================================================================

    async def get_intelligence(self, pipeline_id: str) -> dict[str, Any]:
        """Get per-node intelligence: beliefs, crux status, evidence, precedents."""
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/intelligence",
        )

    async def get_beliefs(self, pipeline_id: str) -> dict[str, Any]:
        """Get belief network state for pipeline nodes."""
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/beliefs",
        )

    async def get_explanations(self, pipeline_id: str) -> dict[str, Any]:
        """Get explainability factors for pipeline decisions."""
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/explanations",
        )

    async def get_precedents(self, pipeline_id: str) -> dict[str, Any]:
        """Get Knowledge Mound precedents for pipeline goals."""
        return await self._client.request(
            "GET",
            f"/api/v1/canvas/pipeline/{pipeline_id}/precedents",
        )

    async def self_improve(
        self,
        pipeline_id: str,
        *,
        budget_limit: float = 10.0,
        require_approval: bool = True,
    ) -> dict[str, Any]:
        """Feed a completed pipeline into the self-improvement system."""
        return await self._client.request(
            "POST",
            f"/api/v1/canvas/pipeline/{pipeline_id}/self-improve",
            json={
                "budget_limit": budget_limit,
                "require_approval": require_approval,
            },
        )

    # =========================================================================
    # Pipeline Agents
    # =========================================================================

    async def get_agents(self, pipeline_id: str) -> dict[str, Any]:
        """Get active agents with status, worktree, and progress."""
        return await self._client.request("GET", f"/api/v1/pipeline/{pipeline_id}/agents")

    async def approve_agent(
        self,
        pipeline_id: str,
        agent_id: str,
        *,
        notes: str = "",
    ) -> dict[str, Any]:
        """Approve an agent's work on a pipeline."""
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/{pipeline_id}/agents/{agent_id}/approve",
            json={"notes": notes},
        )

    async def reject_agent(
        self,
        pipeline_id: str,
        agent_id: str,
        *,
        feedback: str,
    ) -> dict[str, Any]:
        """Reject an agent's work on a pipeline with feedback."""
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/{pipeline_id}/agents/{agent_id}/reject",
            json={"feedback": feedback},
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

    # =========================================================================
    # Pipeline Transition Helpers
    # =========================================================================

    async def ideas_to_goals(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run idea-to-goals transition helper."""
        return await self._client.request(
            "POST",
            "/api/v1/pipeline/transitions/ideas-to-goals",
            json=body,
        )

    async def goals_to_tasks(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run goals-to-tasks transition helper."""
        return await self._client.request(
            "POST",
            "/api/v1/pipeline/transitions/goals-to-tasks",
            json=body,
        )

    async def tasks_to_workflow(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run tasks-to-workflow transition helper."""
        return await self._client.request(
            "POST",
            "/api/v1/pipeline/transitions/tasks-to-workflow",
            json=body,
        )

    async def execute_transitions(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a transition plan."""
        return await self._client.request(
            "POST",
            "/api/v1/pipeline/transitions/execute",
            json=body,
        )

    async def get_transition_provenance(self, node_id: str) -> dict[str, Any]:
        """Get provenance for a transition node."""
        return await self._client.request(
            "GET",
            f"/api/v1/pipeline/transitions/{node_id}/provenance",
        )

    # =========================================================================
    # Universal Pipeline Graph CRUD
    # =========================================================================

    async def get_graph_by_id(self, graph_id: str) -> dict[str, Any]:
        """Get a specific universal pipeline graph."""
        return await self._client.request("GET", f"/api/v1/pipeline/graph/{graph_id}")

    async def delete_graph_by_id(self, graph_id: str) -> dict[str, Any]:
        """Delete a universal pipeline graph."""
        return await self._client.request("DELETE", f"/api/v1/pipeline/graph/{graph_id}")

    async def add_graph_node(
        self,
        graph_id: str,
        node_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Add a node to a universal pipeline graph."""
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/graph/{graph_id}/node",
            json=node_data,
        )

    async def update_graph_node(
        self,
        graph_id: str,
        node_id: str,
        node_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a node in a universal pipeline graph."""
        return await self._client.request(
            "PUT",
            f"/api/v1/pipeline/graph/{graph_id}/node/{node_id}",
            json=node_data,
        )

    async def delete_graph_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Remove a node from a universal pipeline graph."""
        return await self._client.request(
            "DELETE",
            f"/api/v1/pipeline/graph/{graph_id}/node/{node_id}",
        )

    async def reassign_graph_node(
        self,
        graph_id: str,
        node_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """Reassign an agent on a pipeline graph node."""
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/graph/{graph_id}/node/{node_id}/reassign",
            json={"agent_id": agent_id},
        )

    async def list_graph_nodes(
        self,
        graph_id: str,
        *,
        stage: str | None = None,
        subtype: str | None = None,
    ) -> dict[str, Any]:
        """Query nodes in a pipeline graph with optional filters."""
        params: dict[str, Any] = {}
        if stage:
            params["stage"] = stage
        if subtype:
            params["subtype"] = subtype
        return await self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/nodes",
            params=params,
        )

    async def promote_graph_nodes(
        self,
        graph_id: str,
        node_ids: list[str],
        *,
        target_stage: str | None = None,
    ) -> dict[str, Any]:
        """Promote nodes to the next pipeline stage."""
        payload: dict[str, Any] = {"node_ids": node_ids}
        if target_stage:
            payload["target_stage"] = target_stage
        return await self._client.request(
            "POST",
            f"/api/v1/pipeline/graph/{graph_id}/promote",
            json=payload,
        )

    async def get_graph_provenance(self, graph_id: str, node_id: str) -> dict[str, Any]:
        """Get the provenance chain for a specific graph node."""
        return await self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/provenance/{node_id}",
        )

    async def get_graph_react_flow(self, graph_id: str) -> dict[str, Any]:
        """Export a pipeline graph as React Flow JSON."""
        return await self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/react-flow",
        )

    async def get_graph_integrity(self, graph_id: str) -> dict[str, Any]:
        """Get the integrity hash for a pipeline graph."""
        return await self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/integrity",
        )

    async def get_graph_suggestions(self, graph_id: str) -> dict[str, Any]:
        """Get transition suggestions for a pipeline graph."""
        return await self._client.request(
            "GET",
            f"/api/v1/pipeline/graph/{graph_id}/suggestions",
        )

"""
Idea-to-Execution Pipeline.

Orchestrates the full four-stage flow:
  Stage 1 (Ideas) → Stage 2 (Goals) → Stage 3 (Actions) → Stage 4 (Orchestration)

Each transition is a pipeline stage with:
- AI-generated best-effort output
- Human-in-the-loop gate for review/modification
- Provenance chain linking every output to its origins
- SHA-256 content hashes for integrity verification

Usage:
    pipeline = IdeaToExecutionPipeline()

    # From debate
    result = pipeline.from_debate(cartographer_data)

    # From raw ideas
    result = pipeline.from_ideas(["idea 1", "idea 2", ...])

    # Access each stage
    result.ideas_canvas      # Stage 1
    result.goal_graph         # Stage 2
    result.workflow            # Stage 3
    result.execution_plan      # Stage 4
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aragora.canvas.converters import (
    debate_to_ideas_canvas,
    execution_to_orchestration_canvas,
    goals_to_canvas,
    to_react_flow,
    workflow_to_actions_canvas,
)
from aragora.canvas.models import Canvas
from aragora.canvas.stages import (
    PipelineStage,
    ProvenanceLink,
    StageTransition,
    content_hash,
)
from aragora.goals.extractor import GoalExtractionConfig, GoalExtractor, GoalGraph, GoalNode

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for async pipeline execution."""

    stages_to_run: list[str] = field(
        default_factory=lambda: ["ideation", "goals", "workflow", "orchestration"]
    )
    debate_rounds: int = 3
    goal_extraction_config: GoalExtractionConfig | None = None
    workflow_mode: str = "quick"  # "quick" or "debate"
    orchestration_tracks: list[str] | None = None
    max_orchestration_cycles: int = 5
    dry_run: bool = False
    enable_receipts: bool = True
    event_callback: Any | None = None  # callable(event_type: str, data: dict)
    worktree_path: str | None = None  # Git worktree for agent execution
    enable_smart_goals: bool = True
    enable_elo_assignment: bool = True
    enable_km_precedents: bool = True


@dataclass
class StageResult:
    """Result of a single pipeline stage."""

    stage_name: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    output: Any = None
    duration: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "stage_name": self.stage_name,
            "status": self.status,
            "duration": self.duration,
        }
        if self.error:
            result["error"] = self.error
        if self.output is not None and hasattr(self.output, "to_dict"):
            result["output_summary"] = {"type": type(self.output).__name__}
        return result


@dataclass
class PipelineResult:
    """Complete result of the idea-to-execution pipeline.

    Contains the output of each stage as both structured data
    and React Flow-compatible canvas representations.
    """

    pipeline_id: str
    # Stage outputs
    ideas_canvas: Canvas | None = None
    goal_graph: GoalGraph | None = None
    actions_canvas: Canvas | None = None
    orchestration_canvas: Canvas | None = None
    # Stage transitions
    transitions: list[StageTransition] = field(default_factory=list)
    # Full provenance chain
    provenance: list[ProvenanceLink] = field(default_factory=list)
    # Human review status per stage
    stage_status: dict[str, str] = field(default_factory=dict)
    # Async pipeline fields
    stage_results: list[StageResult] = field(default_factory=list)
    final_workflow: dict[str, Any] | None = None
    orchestration_result: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    duration: float = 0.0
    # Universal graph (populated when use_universal=True)
    universal_graph: Any | None = None  # UniversalGraph

    def to_dict(self) -> dict[str, Any]:
        result = {
            "pipeline_id": self.pipeline_id,
            "ideas": to_react_flow(self.ideas_canvas) if self.ideas_canvas else None,
            "goals": self.goal_graph.to_dict() if self.goal_graph else None,
            "actions": to_react_flow(self.actions_canvas) if self.actions_canvas else None,
            "orchestration": (
                to_react_flow(self.orchestration_canvas)
                if self.orchestration_canvas
                else None
            ),
            "transitions": [t.to_dict() for t in self.transitions],
            "provenance": [p.to_dict() for p in self.provenance],
            "provenance_count": len(self.provenance),
            "stage_status": self.stage_status,
            "integrity_hash": self._compute_integrity_hash(),
        }
        if self.stage_results:
            result["stage_results"] = [sr.to_dict() for sr in self.stage_results]
        if self.final_workflow is not None:
            result["final_workflow"] = self.final_workflow
        if self.orchestration_result is not None:
            result["orchestration_result"] = self.orchestration_result
        if self.receipt is not None:
            result["receipt"] = self.receipt
        if self.duration > 0:
            result["duration"] = self.duration
        if self.universal_graph is not None and hasattr(self.universal_graph, "to_dict"):
            result["universal_graph"] = self.universal_graph.to_dict()
        return result

    def _compute_integrity_hash(self) -> str:
        """Compute a pipeline-wide integrity hash."""
        parts = []
        for link in self.provenance:
            parts.append(link.content_hash)
        combined = ":".join(sorted(parts))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


class IdeaToExecutionPipeline:
    """Orchestrates the four-stage idea-to-execution flow.

    Each stage produces a Canvas with typed nodes and provenance links.
    The pipeline can be run end-to-end or stage-by-stage with human
    review gates between stages.
    """

    def __init__(
        self,
        goal_extractor: GoalExtractor | None = None,
        agent: Any | None = None,
        use_universal: bool = False,
    ):
        """Initialize the pipeline.

        Args:
            goal_extractor: Custom GoalExtractor (defaults to structural mode)
            agent: Optional AI agent for synthesis across stages
            use_universal: If True, build a UniversalGraph alongside Canvas outputs
        """
        self._goal_extractor = goal_extractor or GoalExtractor(agent=agent)
        self._agent = agent
        self._use_universal = use_universal

    @classmethod
    def from_demo(cls) -> PipelineResult:
        """Create a pre-built demo pipeline with example data.

        Useful for frontend development and demonstrations.
        """
        pipeline = cls()
        ideas = [
            "Build a rate limiter for API endpoints",
            "Add Redis-backed caching for frequently accessed data",
            "Improve API docs with OpenAPI interactive playground",
            "Set up end-to-end performance monitoring",
        ]
        return pipeline.from_ideas(ideas, auto_advance=True)

    def from_debate(
        self,
        cartographer_data: dict[str, Any],
        auto_advance: bool = True,
        pipeline_id: str | None = None,
        event_callback: Any | None = None,
    ) -> PipelineResult:
        """Run the full pipeline starting from a debate graph.

        Args:
            cartographer_data: ArgumentCartographer.to_dict() output
            auto_advance: If True, auto-generate all stages.
                          If False, stop after Stage 1 for human review.
            pipeline_id: Optional external pipeline ID (generated if not provided)
            event_callback: Optional callable(event_type, data) for progress events
        """
        pipeline_id = pipeline_id or f"pipe-{uuid.uuid4().hex[:8]}"
        result = PipelineResult(
            pipeline_id=pipeline_id,
            stage_status={s.value: "pending" for s in PipelineStage},
        )

        # Stage 1: Ideas
        result.ideas_canvas = debate_to_ideas_canvas(
            cartographer_data,
            canvas_name=f"Ideas from Debate",
        )
        result.stage_status[PipelineStage.IDEAS.value] = "complete"
        self._emit_sync(event_callback, "stage_completed", {"stage": "ideas"})
        logger.info(
            "Pipeline %s: Stage 1 complete — %d idea nodes",
            pipeline_id,
            len(result.ideas_canvas.nodes),
        )

        if not auto_advance:
            return result

        # Stage 2: Goals
        self._emit_sync(event_callback, "stage_started", {"stage": "goals"})
        result = self._advance_to_goals(result)
        self._emit_sync(event_callback, "stage_completed", {"stage": "goals"})
        # Emit node-level events for goals
        if result.goal_graph:
            for goal in result.goal_graph.goals:
                self._emit_sync(event_callback, "pipeline_node_added", {
                    "stage": "goals",
                    "node_id": goal.id,
                    "node_type": goal.goal_type.value,
                    "label": goal.title,
                })

        # Stage 3: Actions
        self._emit_sync(event_callback, "stage_started", {"stage": "actions"})
        result = self._advance_to_actions(result)
        self._emit_sync(event_callback, "stage_completed", {"stage": "actions"})
        # Emit node-level events for action steps
        if result.actions_canvas:
            for node_id, node in result.actions_canvas.nodes.items():
                self._emit_sync(event_callback, "pipeline_node_added", {
                    "stage": "actions",
                    "node_id": node_id,
                    "node_type": node.data.get("step_type", "task"),
                    "label": node.label,
                })

        # Stage 4: Orchestration
        self._emit_sync(event_callback, "stage_started", {"stage": "orchestration"})
        result = self._advance_to_orchestration(result)
        self._emit_sync(event_callback, "stage_completed", {"stage": "orchestration"})
        # Emit node-level events for orchestration tasks
        if result.orchestration_canvas:
            for node_id, node in result.orchestration_canvas.nodes.items():
                self._emit_sync(event_callback, "pipeline_node_added", {
                    "stage": "orchestration",
                    "node_id": node_id,
                    "node_type": node.data.get("orch_type", "agent_task"),
                    "label": node.label,
                })

        self._build_universal_graph(result)
        return result

    def from_ideas(
        self,
        ideas: list[str],
        auto_advance: bool = True,
        pipeline_id: str | None = None,
        event_callback: Any | None = None,
    ) -> PipelineResult:
        """Run the full pipeline from raw idea strings.

        Simpler entry point for users who just have a list of thoughts.

        Args:
            ideas: List of idea/thought strings
            auto_advance: If True, auto-generate all stages
            pipeline_id: Optional external pipeline ID (generated if not provided)
            event_callback: Optional callable(event_type, data) for progress events
        """
        pipeline_id = pipeline_id or f"pipe-{uuid.uuid4().hex[:8]}"
        result = PipelineResult(
            pipeline_id=pipeline_id,
            stage_status={s.value: "pending" for s in PipelineStage},
        )

        # Stage 1: Convert raw ideas to canvas
        result.goal_graph = self._goal_extractor.extract_from_raw_ideas(ideas)
        result.ideas_canvas = Canvas(
            id=f"ideas-raw-{uuid.uuid4().hex[:8]}",
            name="Raw Ideas",
            metadata={"stage": PipelineStage.IDEAS.value, "source": "raw"},
        )
        # Add idea nodes to canvas
        from aragora.canvas.converters import _radial_layout
        from aragora.canvas.models import CanvasNode, CanvasNodeType, Position, Size

        positions = _radial_layout(len(ideas))
        for i, idea in enumerate(ideas):
            pos = positions[i] if i < len(positions) else Position(0, 0)
            node = CanvasNode(
                id=f"raw-idea-{i}",
                node_type=CanvasNodeType.KNOWLEDGE,
                position=pos,
                label=idea[:80],
                data={
                    "stage": PipelineStage.IDEAS.value,
                    "idea_type": "concept",
                    "full_content": idea,
                    "content_hash": content_hash(idea),
                    "rf_type": "ideaNode",
                },
            )
            result.ideas_canvas.nodes[node.id] = node

        result.stage_status[PipelineStage.IDEAS.value] = "complete"
        self._emit_sync(event_callback, "stage_completed", {"stage": "ideas"})
        # Emit node-level events for ideas
        for node_id, node in result.ideas_canvas.nodes.items():
            self._emit_sync(event_callback, "pipeline_node_added", {
                "stage": "ideas",
                "node_id": node_id,
                "node_type": "idea",
                "label": node.label,
            })

        # SMART score goals and detect conflicts
        if result.goal_graph and result.goal_graph.goals:
            try:
                conflicts = self._goal_extractor.detect_goal_conflicts(result.goal_graph)
                if conflicts:
                    result.goal_graph.metadata["conflicts"] = conflicts
            except Exception:
                pass

            for goal in result.goal_graph.goals:
                try:
                    smart_scores = self._goal_extractor.score_smart(goal)
                    goal.metadata["smart_scores"] = smart_scores
                    overall = smart_scores.get("overall", 0.5)
                    if overall >= 0.7:
                        goal.priority = "high"
                    elif overall < 0.4:
                        goal.priority = "low"
                except Exception:
                    pass

        # Goals already extracted via extract_from_raw_ideas
        result.stage_status[PipelineStage.GOALS.value] = "complete"
        self._emit_sync(event_callback, "stage_completed", {"stage": "goals"})
        # Emit node-level events for goals
        if result.goal_graph:
            for goal in result.goal_graph.goals:
                self._emit_sync(event_callback, "pipeline_node_added", {
                    "stage": "goals",
                    "node_id": goal.id,
                    "node_type": goal.goal_type.value,
                    "label": goal.title,
                })

        if not auto_advance:
            return result

        # Stage 3: Actions
        self._emit_sync(event_callback, "stage_started", {"stage": "actions"})
        result = self._advance_to_actions(result)
        self._emit_sync(event_callback, "stage_completed", {"stage": "actions"})
        # Emit node-level events for action steps
        if result.actions_canvas:
            for node_id, node in result.actions_canvas.nodes.items():
                self._emit_sync(event_callback, "pipeline_node_added", {
                    "stage": "actions",
                    "node_id": node_id,
                    "node_type": node.data.get("step_type", "task"),
                    "label": node.label,
                })

        # Stage 4: Orchestration
        self._emit_sync(event_callback, "stage_started", {"stage": "orchestration"})
        result = self._advance_to_orchestration(result)
        self._emit_sync(event_callback, "stage_completed", {"stage": "orchestration"})
        # Emit node-level events for orchestration tasks
        if result.orchestration_canvas:
            for node_id, node in result.orchestration_canvas.nodes.items():
                self._emit_sync(event_callback, "pipeline_node_added", {
                    "stage": "orchestration",
                    "node_id": node_id,
                    "node_type": node.data.get("orch_type", "agent_task"),
                    "label": node.label,
                })

        self._build_universal_graph(result)
        return result

    def advance_stage(
        self,
        result: PipelineResult,
        target_stage: PipelineStage,
    ) -> PipelineResult:
        """Advance the pipeline to a specific stage.

        Used when humans have reviewed and approved a stage,
        and want to advance to the next one.
        """
        if target_stage == PipelineStage.GOALS:
            return self._advance_to_goals(result)
        elif target_stage == PipelineStage.ACTIONS:
            return self._advance_to_actions(result)
        elif target_stage == PipelineStage.ORCHESTRATION:
            return self._advance_to_orchestration(result)
        return result

    # =========================================================================
    # Async pipeline execution
    # =========================================================================

    async def run(
        self,
        input_text: str,
        config: PipelineConfig | None = None,
        pipeline_id: str | None = None,
    ) -> PipelineResult:
        """Run the full async pipeline from input text.

        Executes each configured stage in sequence, emitting events via
        config.event_callback at each stage boundary. Supports dry_run
        mode (skips orchestration) and receipt generation.

        Args:
            input_text: The input idea/question/problem statement
            config: Pipeline configuration
            pipeline_id: Optional external pipeline ID (generated if not provided)

        Returns:
            PipelineResult with all stage outputs
        """
        cfg = config or PipelineConfig()
        pipeline_id = pipeline_id or f"pipe-{uuid.uuid4().hex[:8]}"
        start_time = time.monotonic()

        result = PipelineResult(
            pipeline_id=pipeline_id,
            stage_status={s.value: "pending" for s in PipelineStage},
        )

        self._emit(cfg, "started", {"pipeline_id": pipeline_id, "stages": cfg.stages_to_run})

        try:
            # Stage 1: Ideation
            if "ideation" in cfg.stages_to_run:
                sr = await self._run_ideation(pipeline_id, input_text, cfg)
                result.stage_results.append(sr)
                if sr.status == "completed" and sr.output:
                    result.ideas_canvas = sr.output.get("canvas")
                    result.stage_status[PipelineStage.IDEAS.value] = "complete"
                elif sr.status == "failed":
                    result.stage_status[PipelineStage.IDEAS.value] = "failed"

            # Stage 2: Goal extraction
            if "goals" in cfg.stages_to_run:
                debate_output = (
                    result.stage_results[0].output if result.stage_results else None
                )
                sr = await self._run_goal_extraction(pipeline_id, debate_output, cfg)
                result.stage_results.append(sr)
                if sr.status == "completed" and sr.output:
                    goal_graph = sr.output.get("goal_graph")
                    if goal_graph:
                        result.goal_graph = goal_graph
                        if goal_graph.transition:
                            result.transitions.append(goal_graph.transition)
                        result.provenance.extend(goal_graph.provenance)
                    result.stage_status[PipelineStage.GOALS.value] = "complete"
                elif sr.status == "failed":
                    result.stage_status[PipelineStage.GOALS.value] = "failed"

            # Stage 3: Workflow generation
            if "workflow" in cfg.stages_to_run:
                sr = await self._run_workflow_generation(pipeline_id, result.goal_graph, cfg)
                result.stage_results.append(sr)
                if sr.status == "completed" and sr.output:
                    result.final_workflow = sr.output.get("workflow")
                    # Also advance canvas pipeline
                    if result.goal_graph:
                        result = self._advance_to_actions(result)
                    result.stage_status[PipelineStage.ACTIONS.value] = "complete"
                elif sr.status == "failed":
                    result.stage_status[PipelineStage.ACTIONS.value] = "failed"

            # Stage 4: Orchestration
            if "orchestration" in cfg.stages_to_run:
                if cfg.dry_run:
                    sr = StageResult(stage_name="orchestration", status="skipped")
                    result.stage_results.append(sr)
                else:
                    sr = await self._run_orchestration(
                        pipeline_id, result.final_workflow, result.goal_graph, cfg,
                    )
                    result.stage_results.append(sr)
                    if sr.status == "completed" and sr.output:
                        result.orchestration_result = sr.output.get("orchestration")
                        if result.actions_canvas:
                            result = self._advance_to_orchestration(result)
                        result.stage_status[PipelineStage.ORCHESTRATION.value] = "complete"

            # Generate receipt
            if cfg.enable_receipts and not cfg.dry_run:
                result.receipt = self._generate_receipt(result)

            result.duration = time.monotonic() - start_time
            self._emit(cfg, "completed", {
                "pipeline_id": pipeline_id,
                "duration": result.duration,
                "receipt": result.receipt,
            })

        except Exception as exc:
            result.duration = time.monotonic() - start_time
            logger.warning("Pipeline %s failed: %s", pipeline_id, exc)
            self._emit(cfg, "failed", {
                "pipeline_id": pipeline_id,
                "error": "Pipeline execution failed",
            })

        return result

    async def _run_ideation(
        self,
        pipeline_id: str,
        input_text: str,
        cfg: PipelineConfig,
    ) -> StageResult:
        """Stage 1: Run debate or extract raw ideas."""
        sr = StageResult(stage_name="ideation", status="running")
        start = time.monotonic()
        self._emit(cfg, "stage_started", {"stage": "ideation"})

        try:
            # Try to run a debate for richer ideation
            canvas = None
            debate_data: dict[str, Any] = {}
            try:
                from aragora.debate.orchestrator import Arena
                from aragora.debate.models import DebateProtocol, Environment

                env = Environment(task=input_text)
                protocol = DebateProtocol(rounds=cfg.debate_rounds)
                arena = Arena(env, [], protocol)
                debate_result = await arena.run()
                debate_data = {
                    "debate_result": debate_result,
                    "nodes": getattr(debate_result, "argument_graph", {}).get("nodes", []),
                }
                canvas = debate_to_ideas_canvas(
                    debate_data, canvas_name="Ideas from Debate",
                )
            except (ImportError, Exception):
                # Fallback: extract ideas from raw text
                ideas = [s.strip() for s in input_text.split(".") if s.strip()]
                if not ideas:
                    ideas = [input_text]
                goal_graph = self._goal_extractor.extract_from_raw_ideas(ideas)
                debate_data = {"raw_ideas": ideas, "goal_graph_preview": goal_graph}

                from aragora.canvas.models import Canvas as CanvasModel
                canvas = CanvasModel(
                    id=f"ideas-{uuid.uuid4().hex[:8]}",
                    name="Ideas from Text",
                    metadata={"stage": PipelineStage.IDEAS.value, "source": "text"},
                )

            sr.output = {"canvas": canvas, **debate_data}
            sr.status = "completed"
            sr.duration = time.monotonic() - start
            self._emit(cfg, "stage_completed", {
                "stage": "ideation",
                "summary": {"source": "debate" if "debate_result" in debate_data else "text"},
            })
        except Exception as exc:
            sr.status = "failed"
            sr.error = "Ideation stage failed"
            sr.duration = time.monotonic() - start
            logger.warning("Ideation failed: %s", exc)

        return sr

    async def _run_goal_extraction(
        self,
        pipeline_id: str,
        debate_output: dict[str, Any] | None,
        cfg: PipelineConfig,
    ) -> StageResult:
        """Stage 2: Extract goals from debate analysis."""
        sr = StageResult(stage_name="goals", status="running")
        start = time.monotonic()
        self._emit(cfg, "stage_started", {"stage": "goals"})

        try:
            goal_graph = None

            # If we have debate data with argument nodes, use debate analysis
            if debate_output and debate_output.get("nodes"):
                cartographer_data = {"nodes": debate_output["nodes"]}
                goal_graph = self._goal_extractor.extract_from_debate_analysis(
                    cartographer_data,
                    config=cfg.goal_extraction_config,
                )
            elif debate_output and debate_output.get("goal_graph_preview"):
                # Already extracted via raw ideas
                goal_graph = debate_output["goal_graph_preview"]
            elif debate_output and debate_output.get("canvas"):
                # Use canvas data for structural extraction
                canvas = debate_output["canvas"]
                canvas_data = canvas.to_dict() if hasattr(canvas, "to_dict") else {}
                goal_graph = self._goal_extractor.extract_from_ideas(canvas_data)
            else:
                goal_graph = GoalGraph(id=f"goals-{uuid.uuid4().hex[:8]}")

            # SMART scoring and conflict detection
            if goal_graph and goal_graph.goals:
                # Detect conflicts
                try:
                    conflicts = self._goal_extractor.detect_goal_conflicts(goal_graph)
                    if conflicts:
                        goal_graph.metadata["conflicts"] = conflicts
                except Exception:
                    pass

                # SMART score each goal
                for goal in goal_graph.goals:
                    try:
                        smart_scores = self._goal_extractor.score_smart(goal)
                        goal.metadata["smart_scores"] = smart_scores
                        overall = smart_scores.get("overall", 0.5)
                        if overall >= 0.7:
                            goal.priority = "high"
                        elif overall < 0.4:
                            goal.priority = "low"
                    except Exception:
                        pass

                # Query KM for precedents
                try:
                    from aragora.pipeline.km_bridge import PipelineKMBridge

                    bridge = PipelineKMBridge()
                    if bridge.available:
                        precedents = bridge.query_similar_goals(goal_graph)
                        bridge.enrich_with_precedents(goal_graph, precedents)
                except (ImportError, Exception):
                    pass

            # Emit individual goals
            if goal_graph:
                for goal in goal_graph.goals:
                    self._emit(cfg, "goal_extracted", {"goal": goal.to_dict()})

            sr.output = {"goal_graph": goal_graph}
            sr.status = "completed"
            sr.duration = time.monotonic() - start
            self._emit(cfg, "stage_completed", {
                "stage": "goals",
                "summary": {"goal_count": len(goal_graph.goals) if goal_graph else 0},
            })
        except Exception as exc:
            sr.status = "failed"
            sr.error = "Goal extraction failed"
            sr.duration = time.monotonic() - start
            logger.warning("Goal extraction failed: %s", exc)

        return sr

    async def _run_workflow_generation(
        self,
        pipeline_id: str,
        goal_graph: GoalGraph | None,
        cfg: PipelineConfig,
    ) -> StageResult:
        """Stage 3: Generate workflow from goals."""
        sr = StageResult(stage_name="workflow", status="running")
        start = time.monotonic()
        self._emit(cfg, "stage_started", {"stage": "workflow"})

        try:
            workflow: dict[str, Any] | None = None

            if goal_graph and goal_graph.goals:
                # Try NLWorkflowBuilder
                try:
                    from aragora.workflow.nl_builder import NLWorkflowBuilder

                    builder = NLWorkflowBuilder()
                    goal_texts = [g.title for g in goal_graph.goals]
                    nl_input = ". ".join(goal_texts)

                    if cfg.workflow_mode == "quick":
                        nl_result = builder.build_quick(nl_input)
                    else:
                        nl_result = await builder.build(nl_input)

                    workflow = (
                        nl_result.to_dict()
                        if hasattr(nl_result, "to_dict")
                        else {"steps": [], "name": "generated"}
                    )
                except (ImportError, Exception):
                    # Fallback: use internal goal-to-workflow conversion
                    workflow = self._goals_to_workflow(goal_graph)

                self._emit(cfg, "workflow_generated", {"workflow": workflow})
            else:
                workflow = {"steps": [], "name": "empty"}

            sr.output = {"workflow": workflow}
            sr.status = "completed"
            sr.duration = time.monotonic() - start
            self._emit(cfg, "stage_completed", {
                "stage": "workflow",
                "summary": {"step_count": len(workflow.get("steps", []))},
            })
        except Exception as exc:
            sr.status = "failed"
            sr.error = "Workflow generation failed"
            sr.duration = time.monotonic() - start
            logger.warning("Workflow generation failed: %s", exc)

        return sr

    async def _run_orchestration(
        self,
        pipeline_id: str,
        workflow: dict[str, Any] | None,
        goal_graph: GoalGraph | None,
        cfg: PipelineConfig,
    ) -> StageResult:
        """Stage 4: Run orchestration on workflow.

        Builds an execution plan from the workflow/goal_graph, then executes
        each task using DebugLoop (with graceful fallback). Emits node-level
        events for real-time frontend updates.
        """
        sr = StageResult(stage_name="orchestration", status="running")
        start = time.monotonic()
        self._emit(cfg, "stage_started", {"stage": "orchestration"})

        try:
            execution_plan = self._build_execution_plan(workflow, goal_graph)

            if not execution_plan.get("tasks"):
                sr.output = {"orchestration": {"status": "skipped", "reason": "no tasks"}}
                sr.status = "completed"
                sr.duration = time.monotonic() - start
                return sr

            results: list[dict[str, Any]] = []
            for task in execution_plan["tasks"]:
                if task["type"] == "human_gate":
                    results.append({
                        "task_id": task["id"],
                        "status": "awaiting_approval",
                        "name": task["name"],
                    })
                    self._emit(cfg, "pipeline_node_added", {
                        "stage": "orchestration",
                        "node_id": task["id"],
                        "node_type": "human_gate",
                        "label": task["name"],
                    })
                    continue

                self._emit(cfg, "pipeline_agent_started", {
                    "task_id": task["id"],
                    "name": task["name"],
                })

                task_result = await self._execute_task(task, cfg)
                results.append(task_result)

                self._emit(cfg, "pipeline_agent_completed", {
                    "task_id": task["id"],
                    "status": task_result["status"],
                })
                self._emit(cfg, "pipeline_node_added", {
                    "stage": "orchestration",
                    "node_id": task["id"],
                    "node_type": "agent_task",
                    "label": task["name"],
                })

            completed = sum(1 for r in results if r["status"] == "completed")
            total = len(results)
            orch_result: dict[str, Any] = {
                "status": "executed",
                "tasks_completed": completed,
                "tasks_total": total,
                "results": results,
            }

            sr.output = {"orchestration": orch_result}
            sr.status = "completed"
            sr.duration = time.monotonic() - start
            self._emit(cfg, "stage_completed", {
                "stage": "orchestration",
                "summary": orch_result,
            })
        except Exception as exc:
            sr.status = "failed"
            sr.error = "Orchestration failed"
            sr.duration = time.monotonic() - start
            logger.warning("Orchestration failed: %s", exc)

        return sr

    def _build_execution_plan(
        self,
        workflow: dict[str, Any] | None,
        goal_graph: GoalGraph | None,
    ) -> dict[str, Any]:
        """Build a flat task list from workflow steps or goal graph nodes.

        Extracts tasks from the workflow (preferred) or falls back to the
        goal graph. Each task has an id, name, description, type, and
        optional test_scope.
        """
        tasks: list[dict[str, Any]] = []
        if workflow and workflow.get("steps"):
            for step in workflow["steps"]:
                tasks.append({
                    "id": step["id"],
                    "name": step["name"],
                    "description": step.get("description", ""),
                    "type": (
                        "human_gate"
                        if step.get("step_type") == "human_checkpoint"
                        else "agent_task"
                    ),
                    "test_scope": step.get("config", {}).get("test_scope", []),
                })
        elif goal_graph and goal_graph.goals:
            for goal in goal_graph.goals:
                tasks.append({
                    "id": goal.id,
                    "name": goal.title,
                    "description": goal.description,
                    "type": "agent_task",
                    "test_scope": [],
                })
        return {"tasks": tasks}

    async def _execute_task(
        self,
        task: dict[str, Any],
        cfg: PipelineConfig,
    ) -> dict[str, Any]:
        """Execute a single task, using DebugLoop if available.

        Falls back to ``planned`` status when the execution engine
        (DebugLoop / ClaudeCodeHarness) is not installed or raises.
        """
        try:
            from aragora.nomic.debug_loop import DebugLoop, DebugLoopConfig

            loop_cfg = DebugLoopConfig(max_retries=2)
            loop = DebugLoop(loop_cfg)
            result = await loop.execute_with_retry(
                instruction=(
                    f"Implement: {task['name']}\n\n{task.get('description', '')}"
                ),
                worktree_path=getattr(cfg, "worktree_path", None) or "/tmp/aragora-worktree",
                test_scope=task.get("test_scope", []),
            )
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "completed" if result.success else "failed",
                "output": result.to_dict() if hasattr(result, "to_dict") else {},
            }
        except (ImportError, AttributeError):
            # DebugLoop not available — fall back to planned status
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "planned",
                "output": {"reason": "execution_engine_unavailable"},
            }
        except (RuntimeError, OSError):
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "failed",
                "output": {"error": "Task execution failed"},
            }

    def _emit(self, cfg: PipelineConfig, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event via the configured callback."""
        if cfg.event_callback:
            try:
                cfg.event_callback(event_type, data)
            except Exception:
                pass

    @staticmethod
    def _emit_sync(
        callback: Any | None, event_type: str, data: dict[str, Any],
    ) -> None:
        """Emit an event via a standalone callback (for sync methods)."""
        if callback:
            try:
                callback(event_type, data)
            except Exception:
                pass

    def _generate_receipt(self, result: PipelineResult) -> dict[str, Any] | None:
        """Generate a decision receipt for the completed pipeline.

        Collects participants from orchestration results, evidence from
        each stage, and builds a rich receipt. Falls back to a lightweight
        dict when the gauntlet receipt module is unavailable.
        """
        # Collect participants from orchestration result
        participants: list[str] = []
        if result.orchestration_result and isinstance(result.orchestration_result, dict):
            for task_result in result.orchestration_result.get("results", []):
                name = task_result.get("name", "unknown")
                if name not in participants:
                    participants.append(name)

        # Collect evidence from stage results
        evidence: list[dict[str, Any]] = []
        for sr in result.stage_results:
            evidence.append({
                "stage": sr.stage_name,
                "status": sr.status,
                "duration": sr.duration,
            })

        stages_completed = sum(
            1 for sr in result.stage_results if sr.status == "completed"
        )

        try:
            from aragora.gauntlet.receipts import DecisionReceipt

            receipt = DecisionReceipt(
                decision_id=result.pipeline_id,
                decision_summary=(
                    f"Pipeline completed: {len(result.stage_results)} stages, "
                    f"{len(result.provenance)} provenance links"
                ),
                confidence=result.transitions[-1].confidence if result.transitions else 0.5,
                participants=participants,
                evidence=evidence,
            )
            return receipt.to_dict()
        except (ImportError, Exception):
            return {
                "pipeline_id": result.pipeline_id,
                "integrity_hash": result._compute_integrity_hash(),
                "stages_completed": stages_completed,
                "provenance_count": len(result.provenance),
                "participants": participants,
                "evidence": evidence,
            }

    # =========================================================================
    # Stage transition methods (sync, for from_debate/from_ideas)
    # =========================================================================

    def _advance_to_goals(self, result: PipelineResult) -> PipelineResult:
        """Stage 1 → Stage 2: Extract goals from ideas."""
        if not result.ideas_canvas:
            logger.warning("Cannot advance to goals: no ideas canvas")
            return result

        canvas_data = result.ideas_canvas.to_dict()

        # Pre-cluster related ideas semantically
        try:
            canvas_data = self._goal_extractor.cluster_ideas_semantically(canvas_data)
        except Exception:
            pass  # Continue with unclustered data

        result.goal_graph = self._goal_extractor.extract_from_ideas(canvas_data)

        # Detect conflicts between goals
        try:
            conflicts = self._goal_extractor.detect_goal_conflicts(result.goal_graph)
            if conflicts:
                result.goal_graph.metadata["conflicts"] = conflicts
        except Exception:
            pass

        # SMART score each goal and adjust priority
        for goal in result.goal_graph.goals:
            try:
                smart_scores = self._goal_extractor.score_smart(goal)
                goal.metadata["smart_scores"] = smart_scores
                overall = smart_scores.get("overall", 0.5)
                if overall >= 0.7:
                    goal.priority = "high"
                elif overall < 0.4:
                    goal.priority = "low"
            except Exception:
                pass

        # Query KM for precedents
        try:
            from aragora.pipeline.km_bridge import PipelineKMBridge

            bridge = PipelineKMBridge()
            if bridge.available:
                precedents = bridge.query_similar_goals(result.goal_graph)
                bridge.enrich_with_precedents(result.goal_graph, precedents)
        except (ImportError, Exception):
            pass

        if result.goal_graph.transition:
            result.transitions.append(result.goal_graph.transition)
        result.provenance.extend(result.goal_graph.provenance)
        result.stage_status[PipelineStage.GOALS.value] = "complete"

        logger.info(
            "Pipeline %s: Stage 2 complete — %d goals extracted",
            result.pipeline_id,
            len(result.goal_graph.goals),
        )
        return result

    def _advance_to_actions(self, result: PipelineResult) -> PipelineResult:
        """Stage 2 → Stage 3: Generate workflow from goals."""
        if not result.goal_graph or not result.goal_graph.goals:
            logger.warning("Cannot advance to actions: no goals")
            return result

        # Convert goals into a WorkflowDefinition-like structure
        workflow_data = self._goals_to_workflow(result.goal_graph)

        # Create provenance links
        provenance: list[ProvenanceLink] = []
        for goal in result.goal_graph.goals:
            for step in workflow_data.get("steps", []):
                if step.get("source_goal_id") == goal.id:
                    provenance.append(
                        ProvenanceLink(
                            source_node_id=goal.id,
                            source_stage=PipelineStage.GOALS,
                            target_node_id=step["id"],
                            target_stage=PipelineStage.ACTIONS,
                            content_hash=content_hash(goal.title),
                            method="goal_decomposition",
                        )
                    )

        result.actions_canvas = workflow_to_actions_canvas(
            workflow_data,
            provenance=provenance,
            canvas_name="Action Plan",
        )
        result.provenance.extend(provenance)

        transition = StageTransition(
            id=f"trans-goals-actions-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.GOALS,
            to_stage=PipelineStage.ACTIONS,
            provenance=provenance,
            status="pending",
            confidence=0.7,
            ai_rationale=(
                f"Decomposed {len(result.goal_graph.goals)} goals into "
                f"{len(workflow_data.get('steps', []))} action steps"
            ),
        )
        result.transitions.append(transition)
        result.stage_status[PipelineStage.ACTIONS.value] = "complete"

        logger.info(
            "Pipeline %s: Stage 3 complete — %d action steps",
            result.pipeline_id,
            len(result.actions_canvas.nodes),
        )
        return result

    def _advance_to_orchestration(self, result: PipelineResult) -> PipelineResult:
        """Stage 3 → Stage 4: Create multi-agent execution plan."""
        if not result.actions_canvas:
            logger.warning("Cannot advance to orchestration: no actions canvas")
            return result

        # Build execution plan from action steps
        execution_plan = self._actions_to_execution_plan(result.actions_canvas)

        # Create provenance links
        provenance: list[ProvenanceLink] = []
        for task in execution_plan.get("tasks", []):
            source_id = task.get("source_action_id", "")
            if source_id:
                provenance.append(
                    ProvenanceLink(
                        source_node_id=source_id,
                        source_stage=PipelineStage.ACTIONS,
                        target_node_id=task["id"],
                        target_stage=PipelineStage.ORCHESTRATION,
                        content_hash=content_hash(task.get("name", "")),
                        method="agent_assignment",
                    )
                )

        result.orchestration_canvas = execution_to_orchestration_canvas(
            execution_plan,
            canvas_name="Orchestration Plan",
        )
        result.provenance.extend(provenance)

        transition = StageTransition(
            id=f"trans-actions-orch-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.ACTIONS,
            to_stage=PipelineStage.ORCHESTRATION,
            provenance=provenance,
            status="pending",
            confidence=0.6,
            ai_rationale=(
                f"Assigned {len(execution_plan.get('tasks', []))} tasks "
                f"across {len(execution_plan.get('agents', []))} agents"
            ),
        )
        result.transitions.append(transition)
        result.stage_status[PipelineStage.ORCHESTRATION.value] = "complete"

        logger.info(
            "Pipeline %s: Stage 4 complete — %d agents, %d tasks",
            result.pipeline_id,
            len(execution_plan.get("agents", [])),
            len(execution_plan.get("tasks", [])),
        )
        return result

    # =========================================================================
    # Universal graph integration
    # =========================================================================

    def _build_universal_graph(self, result: PipelineResult) -> None:
        """Build a UniversalGraph from completed pipeline stages."""
        if not self._use_universal:
            return
        try:
            from aragora.pipeline.adapters import (
                canvas_to_universal_graph,
                from_goal_node,
            )
            from aragora.pipeline.universal_node import UniversalGraph

            graph = UniversalGraph(
                id=f"ugraph-{result.pipeline_id}",
                name=f"Pipeline {result.pipeline_id}",
            )

            # Stage 1: Ideas
            if result.ideas_canvas:
                ideas_ug = canvas_to_universal_graph(
                    result.ideas_canvas, PipelineStage.IDEAS
                )
                for node in ideas_ug.nodes.values():
                    graph.add_node(node)
                for edge in ideas_ug.edges.values():
                    graph.edges[edge.id] = edge

            # Stage 2: Goals
            if result.goal_graph:
                for goal in result.goal_graph.goals:
                    unode = from_goal_node(goal)
                    graph.add_node(unode)

            # Stage 3: Actions
            if result.actions_canvas:
                actions_ug = canvas_to_universal_graph(
                    result.actions_canvas, PipelineStage.ACTIONS
                )
                for node in actions_ug.nodes.values():
                    graph.add_node(node)
                for edge in actions_ug.edges.values():
                    graph.edges[edge.id] = edge

            # Stage 4: Orchestration
            if result.orchestration_canvas:
                orch_ug = canvas_to_universal_graph(
                    result.orchestration_canvas, PipelineStage.ORCHESTRATION
                )
                for node in orch_ug.nodes.values():
                    graph.add_node(node)
                for edge in orch_ug.edges.values():
                    graph.edges[edge.id] = edge

            graph.transitions = list(result.transitions)
            result.universal_graph = graph
        except Exception as exc:
            logger.warning("Failed to build universal graph: %s", exc)

    # =========================================================================
    # Conversion helpers
    # =========================================================================

    def _goals_to_workflow(self, goal_graph: GoalGraph) -> dict[str, Any]:
        """Convert goals into a workflow definition.

        Each goal decomposes into multiple workflow steps based on type:
        - Goals → research + implement + test + review
        - Milestones → checkpoint + verification
        - Principles → define + validate
        - Strategies → research + design + implement
        - Metrics → instrument + baseline + monitor
        - Risks → assess + mitigate + verify
        """
        steps: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []

        # Decomposition templates by goal type
        decomposition = {
            "goal": [
                ("research", "task", "Research: {title}"),
                ("implement", "task", "Implement: {title}"),
                ("test", "verification", "Test: {title}"),
                ("review", "human_checkpoint", "Review: {title}"),
            ],
            "milestone": [
                ("checkpoint", "human_checkpoint", "Checkpoint: {title}"),
                ("verify", "verification", "Verify: {title}"),
            ],
            "principle": [
                ("define", "task", "Define: {title}"),
                ("validate", "verification", "Validate: {title}"),
            ],
            "strategy": [
                ("research", "task", "Research: {title}"),
                ("design", "task", "Design: {title}"),
                ("implement", "task", "Implement: {title}"),
            ],
            "metric": [
                ("instrument", "task", "Instrument: {title}"),
                ("baseline", "verification", "Baseline: {title}"),
                ("monitor", "task", "Monitor: {title}"),
            ],
            "risk": [
                ("assess", "task", "Assess: {title}"),
                ("mitigate", "task", "Mitigate: {title}"),
                ("verify", "verification", "Verify mitigation: {title}"),
            ],
        }

        for goal in goal_graph.goals:
            template = decomposition.get(goal.goal_type.value, [
                ("execute", "task", "{title}"),
            ])

            goal_step_ids: list[str] = []
            for phase, step_type, name_fmt in template:
                step_id = f"step-{goal.id}-{phase}"
                step = {
                    "id": step_id,
                    "name": name_fmt.format(title=goal.title),
                    "description": goal.description,
                    "step_type": step_type,
                    "source_goal_id": goal.id,
                    "phase": phase,
                    "config": {
                        "priority": goal.priority,
                        "measurable": goal.measurable,
                    },
                    "timeout_seconds": 3600,
                    "retries": 1,
                    "optional": goal.priority == "low",
                }
                steps.append(step)

                # Chain phases within a goal sequentially
                if goal_step_ids:
                    transitions.append({
                        "id": f"seq-{goal_step_ids[-1]}-{step_id}",
                        "from_step": goal_step_ids[-1],
                        "to_step": step_id,
                        "condition": "",
                        "label": "then",
                        "priority": 0,
                    })

                goal_step_ids.append(step_id)

            # Create transitions from goal dependencies (link last step of dep
            # to first step of this goal)
            for dep_goal_id in goal.dependencies:
                dep_steps = [
                    s for s in steps if s.get("source_goal_id") == dep_goal_id
                ]
                if dep_steps and goal_step_ids:
                    transitions.append({
                        "id": f"dep-{dep_goal_id}-{goal.id}",
                        "from_step": dep_steps[-1]["id"],
                        "to_step": goal_step_ids[0],
                        "condition": "",
                        "label": "after",
                        "priority": 0,
                    })

        # Chain independent goal groups sequentially (link last step of one
        # to first step of next, only for goals with no explicit dependencies)
        prev_last_step: str | None = None
        for goal in goal_graph.goals:
            g_steps = [s for s in steps if s.get("source_goal_id") == goal.id]
            if not g_steps:
                continue
            first_id = g_steps[0]["id"]
            last_id = g_steps[-1]["id"]
            has_dep = any(t["to_step"] == first_id for t in transitions)
            if not has_dep and prev_last_step:
                transitions.append({
                    "id": f"seq-{prev_last_step}-{first_id}",
                    "from_step": prev_last_step,
                    "to_step": first_id,
                    "condition": "",
                    "label": "then",
                    "priority": 0,
                })
            prev_last_step = last_id

        return {
            "id": f"wf-{goal_graph.id}",
            "name": "Goal Implementation Workflow",
            "steps": steps,
            "transitions": transitions,
            "entry_step": steps[0]["id"] if steps else None,
        }

    def _actions_to_execution_plan(
        self, actions_canvas: Canvas
    ) -> dict[str, Any]:
        """Convert action canvas nodes into a multi-agent execution plan.

        Assigns tasks to specialized agents based on step phase and type,
        using Aragora's agent archetypes for heterogeneous model consensus.
        When available, uses ELO-aware TeamSelector for data-driven assignment.
        """
        # Try ELO-aware agent ranking
        elo_scores: dict[str, float] = {}
        _PHASE_TO_DOMAIN = {
            "research": "research",
            "design": "creative",
            "implement": "code",
            "test": "technical",
            "verify": "technical",
            "review": "reasoning",
            "define": "research",
            "validate": "technical",
            "assess": "research",
            "mitigate": "code",
            "instrument": "code",
            "baseline": "technical",
            "monitor": "technical",
            "execute": "code",
        }
        _elo_domain_agents: dict[str, str] = {}
        try:
            from aragora.debate.team_selector import TeamSelector

            selector = TeamSelector()
            # Query ELO rankings for each domain
            for domain in set(_PHASE_TO_DOMAIN.values()):
                try:
                    rankings = selector.get_rankings(domain=domain) if hasattr(selector, "get_rankings") else []
                    if rankings:
                        best = rankings[0]
                        agent_id = getattr(best, "agent_id", None) or str(best)
                        score = getattr(best, "elo", 1000.0)
                        elo_scores[f"{domain}:{agent_id}"] = score
                        _elo_domain_agents[domain] = agent_id
                except Exception:
                    pass
        except (ImportError, Exception):
            pass  # Fall back to static map

        # Agent pool with diverse model providers for adversarial vetting
        agents = [
            {
                "id": "agent-researcher",
                "name": "Researcher",
                "type": "anthropic-api",
                "capabilities": ["research", "analysis", "synthesis"],
            },
            {
                "id": "agent-designer",
                "name": "Designer",
                "type": "openai-api",
                "capabilities": ["design", "architecture", "planning"],
            },
            {
                "id": "agent-implementer",
                "name": "Implementer",
                "type": "codex",
                "capabilities": ["code", "implementation", "debugging"],
            },
            {
                "id": "agent-tester",
                "name": "Tester",
                "type": "anthropic-api",
                "capabilities": ["testing", "verification", "validation"],
            },
            {
                "id": "agent-reviewer",
                "name": "Reviewer",
                "type": "gemini",
                "capabilities": ["review", "critique", "quality"],
            },
            {
                "id": "agent-monitor",
                "name": "Monitor",
                "type": "mistral",
                "capabilities": ["monitoring", "metrics", "observability"],
            },
        ]

        # Map step phases to best-fit agent
        phase_agent_map = {
            "research": "agent-researcher",
            "design": "agent-designer",
            "implement": "agent-implementer",
            "test": "agent-tester",
            "verify": "agent-tester",
            "review": "agent-reviewer",
            "checkpoint": "",  # human gate
            "define": "agent-researcher",
            "validate": "agent-tester",
            "assess": "agent-researcher",
            "mitigate": "agent-implementer",
            "instrument": "agent-implementer",
            "baseline": "agent-tester",
            "monitor": "agent-monitor",
            "execute": "agent-implementer",
        }

        tasks: list[dict[str, Any]] = []
        used_agents: set[str] = set()

        for node_id, node in actions_canvas.nodes.items():
            step_type = node.data.get("step_type", "task")
            phase = node.data.get("phase", "")

            # Determine assignment: prefer ELO-ranked agent, fall back to static map
            elo_agent = None
            elo_score = None
            if _elo_domain_agents and phase:
                domain = _PHASE_TO_DOMAIN.get(phase)
                if domain and domain in _elo_domain_agents:
                    elo_agent = _elo_domain_agents[domain]
                    elo_score = elo_scores.get(f"{domain}:{elo_agent}")

            if elo_agent:
                assigned = elo_agent
            elif phase and phase in phase_agent_map:
                assigned = phase_agent_map[phase]
            elif step_type == "human_checkpoint":
                assigned = ""
            elif step_type == "verification":
                assigned = "agent-tester"
            elif step_type == "task":
                assigned = "agent-implementer"
            else:
                assigned = "agent-researcher"

            # Determine task type
            if step_type == "human_checkpoint":
                task_type = "human_gate"
                assigned = ""
            elif step_type == "verification":
                task_type = "verification"
            else:
                task_type = "agent_task"

            if assigned:
                used_agents.add(assigned)

            # Find dependencies from canvas edges
            deps = [
                edge.source_id
                for edge in actions_canvas.edges.values()
                if edge.target_id == node_id
            ]

            task_dict: dict[str, Any] = {
                "id": f"exec-{node_id}",
                "name": node.label,
                "type": task_type,
                "assigned_agent": assigned,
                "depends_on": [f"exec-{d}" for d in deps],
                "source_action_id": node_id,
            }
            if elo_score is not None:
                task_dict["elo_score"] = elo_score
            tasks.append(task_dict)

        # Only include agents that are actually assigned tasks
        active_agents = [a for a in agents if a["id"] in used_agents]

        return {"agents": active_agents, "tasks": tasks}

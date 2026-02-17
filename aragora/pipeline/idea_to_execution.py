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
from aragora.goals.extractor import GoalExtractor, GoalGraph, GoalNode

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "provenance_count": len(self.provenance),
            "stage_status": self.stage_status,
            "integrity_hash": self._compute_integrity_hash(),
        }

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
    ):
        """Initialize the pipeline.

        Args:
            goal_extractor: Custom GoalExtractor (defaults to structural mode)
            agent: Optional AI agent for synthesis across stages
        """
        self._goal_extractor = goal_extractor or GoalExtractor(agent=agent)
        self._agent = agent

    def from_debate(
        self,
        cartographer_data: dict[str, Any],
        auto_advance: bool = True,
    ) -> PipelineResult:
        """Run the full pipeline starting from a debate graph.

        Args:
            cartographer_data: ArgumentCartographer.to_dict() output
            auto_advance: If True, auto-generate all stages.
                          If False, stop after Stage 1 for human review.
        """
        pipeline_id = f"pipe-{uuid.uuid4().hex[:8]}"
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
        logger.info(
            "Pipeline %s: Stage 1 complete — %d idea nodes",
            pipeline_id,
            len(result.ideas_canvas.nodes),
        )

        if not auto_advance:
            return result

        # Stage 2: Goals
        result = self._advance_to_goals(result)

        # Stage 3: Actions
        result = self._advance_to_actions(result)

        # Stage 4: Orchestration
        result = self._advance_to_orchestration(result)

        return result

    def from_ideas(
        self,
        ideas: list[str],
        auto_advance: bool = True,
    ) -> PipelineResult:
        """Run the full pipeline from raw idea strings.

        Simpler entry point for users who just have a list of thoughts.
        """
        pipeline_id = f"pipe-{uuid.uuid4().hex[:8]}"
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

        # Goals already extracted via extract_from_raw_ideas
        result.stage_status[PipelineStage.GOALS.value] = "complete"

        if not auto_advance:
            return result

        # Stage 3: Actions
        result = self._advance_to_actions(result)

        # Stage 4: Orchestration
        result = self._advance_to_orchestration(result)

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
    # Stage transition methods
    # =========================================================================

    def _advance_to_goals(self, result: PipelineResult) -> PipelineResult:
        """Stage 1 → Stage 2: Extract goals from ideas."""
        if not result.ideas_canvas:
            logger.warning("Cannot advance to goals: no ideas canvas")
            return result

        canvas_data = result.ideas_canvas.to_dict()
        result.goal_graph = self._goal_extractor.extract_from_ideas(canvas_data)

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
    # Conversion helpers
    # =========================================================================

    def _goals_to_workflow(self, goal_graph: GoalGraph) -> dict[str, Any]:
        """Convert goals into a workflow definition.

        Each goal becomes one or more workflow steps:
        - Goals → task steps (the work to do)
        - Milestones → checkpoint steps (verification points)
        - Principles → constraint validation steps
        - Risks → verification steps
        """
        steps: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []

        for i, goal in enumerate(goal_graph.goals):
            step_type = {
                "goal": "task",
                "milestone": "human_checkpoint",
                "principle": "verification",
                "strategy": "task",
                "metric": "verification",
                "risk": "verification",
            }.get(goal.goal_type.value, "task")

            step = {
                "id": f"step-{goal.id}",
                "name": goal.title,
                "description": goal.description,
                "step_type": step_type,
                "source_goal_id": goal.id,
                "config": {
                    "priority": goal.priority,
                    "measurable": goal.measurable,
                },
                "timeout_seconds": 3600,
                "retries": 1,
                "optional": goal.priority == "low",
            }
            steps.append(step)

            # Create transitions from dependencies
            for dep_goal_id in goal.dependencies:
                transitions.append({
                    "id": f"trans-{dep_goal_id}-{goal.id}",
                    "from_step": f"step-{dep_goal_id}",
                    "to_step": f"step-{goal.id}",
                    "condition": "",
                    "label": "after",
                    "priority": 0,
                })

        # Chain non-dependent steps sequentially
        independent = [s for s in steps if not any(
            t["to_step"] == s["id"] for t in transitions
        )]
        for i in range(len(independent) - 1):
            transitions.append({
                "id": f"seq-{independent[i]['id']}-{independent[i+1]['id']}",
                "from_step": independent[i]["id"],
                "to_step": independent[i + 1]["id"],
                "condition": "",
                "label": "then",
                "priority": 0,
            })

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

        Assigns tasks to agents based on step type:
        - task → general agent (Claude/GPT)
        - verification → review agent
        - human_checkpoint → human gate (no agent)
        - implementation → code agent (Codex/Claude)
        """
        # Define available agent archetypes
        agents = [
            {
                "id": "agent-analyst",
                "name": "Analyst",
                "type": "claude",
                "capabilities": ["research", "analysis", "synthesis"],
            },
            {
                "id": "agent-implementer",
                "name": "Implementer",
                "type": "codex",
                "capabilities": ["code", "implementation", "debugging"],
            },
            {
                "id": "agent-reviewer",
                "name": "Reviewer",
                "type": "claude",
                "capabilities": ["review", "verification", "testing"],
            },
        ]

        tasks: list[dict[str, Any]] = []
        for node_id, node in actions_canvas.nodes.items():
            step_type = node.data.get("step_type", "task")

            # Assign agent based on step type
            if step_type in ("verification", "human_checkpoint"):
                assigned = "agent-reviewer"
                task_type = "verification"
            elif step_type == "implementation":
                assigned = "agent-implementer"
                task_type = "agent_task"
            else:
                assigned = "agent-analyst"
                task_type = "agent_task"

            if step_type == "human_checkpoint":
                task_type = "human_gate"
                assigned = ""

            # Find dependencies from canvas edges
            deps = [
                edge.source_id
                for edge in actions_canvas.edges.values()
                if edge.target_id == node_id
            ]

            tasks.append({
                "id": f"exec-{node_id}",
                "name": node.label,
                "type": task_type,
                "assigned_agent": assigned,
                "depends_on": [f"exec-{d}" for d in deps],
                "source_action_id": node_id,
            })

        return {"agents": agents, "tasks": tasks}

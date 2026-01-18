"""
Sequential Pattern - Linear agent pipeline with data passing.

The Sequential pattern executes agents one after another, where each agent
builds upon the output of the previous one. This is ideal for:
- Multi-step analysis pipelines
- Document processing workflows
- Iterative refinement chains

Structure:
    [Input] -> [Agent 1] -> [Agent 2] -> [Agent 3] -> [Output]

Configuration:
    - agents: Ordered list of agents to execute
    - prompts: Per-agent prompts (or single prompt for all)
    - pass_context: Whether to pass full context or just previous output
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aragora.workflow.types import (
    WorkflowDefinition,
    Position,
    NodeCategory,
    WorkflowCategory,
)
from aragora.workflow.patterns.base import WorkflowPattern, PatternType


class SequentialPattern(WorkflowPattern):
    """
    Linear agent pipeline with data passing.

    Agents execute in sequence, each building on the previous output.
    Supports per-agent prompts or a common task prompt.

    Example:
        workflow = SequentialPattern.create(
            name="Document Analysis Pipeline",
            agents=["claude", "gpt4"],
            prompts={
                "claude": "Extract key facts from: {input}",
                "gpt4": "Analyze the extracted facts: {step.claude}",
            },
        )

        # Or with roles
        workflow = SequentialPattern.create(
            name="Code Review Pipeline",
            stages=[
                {"agent": "claude", "role": "security_reviewer", "focus": "security"},
                {"agent": "gpt4", "role": "performance_reviewer", "focus": "performance"},
                {"agent": "claude", "role": "synthesizer", "focus": "final_review"},
            ],
            task="Review this code: {code}",
        )
    """

    pattern_type = PatternType.SEQUENTIAL

    def __init__(
        self,
        name: str,
        agents: Optional[List[str]] = None,
        task: str = "",
        prompts: Optional[Dict[str, str]] = None,
        stages: Optional[List[Dict[str, Any]]] = None,
        pass_full_context: bool = True,
        timeout_per_step: float = 120.0,
        **kwargs,
    ):
        super().__init__(name, agents, task, **kwargs)
        self.prompts = prompts or {}
        self.stages = stages
        self.pass_full_context = pass_full_context
        self.timeout_per_step = timeout_per_step

    def create_workflow(self) -> WorkflowDefinition:
        """Create a sequential workflow definition."""
        workflow_id = self._generate_id("seq")
        steps = []
        transitions = []

        # Use stages if provided, otherwise build from agents
        if self.stages:
            stage_configs = self.stages
        else:
            stage_configs = [
                {"agent": agent, "role": agent, "prompt": self.prompts.get(agent, self.task)}
                for agent in self.agents
            ]

        # Calculate positions for visual layout
        start_x = 100
        y_pos = 200
        spacing = 250

        prev_step_id = None
        for i, stage in enumerate(stage_configs):
            agent_type = stage.get("agent", self.agents[0] if self.agents else "claude")
            role = stage.get("role", agent_type)
            prompt = stage.get("prompt", self.task)
            focus = stage.get("focus", "")

            step_id = f"stage_{i}_{role}"

            # Build prompt with context from previous step
            if prev_step_id and self.pass_full_context:
                prompt = self._build_context_prompt(prompt, prev_step_id, focus)

            step = self._create_agent_step(
                step_id=step_id,
                name=f"Stage {i+1}: {role.replace('_', ' ').title()}",
                agent_type=agent_type,
                prompt=prompt,
                position=Position(x=start_x + i * spacing, y=y_pos),
                timeout=self.timeout_per_step,
            )

            if focus:
                step.config["focus"] = focus

            steps.append(step)

            # Create transition from previous step
            if prev_step_id:
                transitions.append(self._create_transition(
                    from_step=prev_step_id,
                    to_step=step_id,
                ))
                # Also set next_steps for sequential fallback
                for s in steps:
                    if s.id == prev_step_id:
                        s.next_steps = [step_id]

            prev_step_id = step_id

        # Create final output aggregation step
        if len(steps) > 1:
            output_step = self._create_task_step(
                step_id="output",
                name="Final Output",
                task_type="transform",
                config={
                    "transform": f"outputs.get('{prev_step_id}', {{}})",
                    "output_format": "json",
                },
                position=Position(x=start_x + len(stage_configs) * spacing, y=y_pos),
                category=NodeCategory.TASK,
            )
            steps.append(output_step)

            if prev_step_id:
                transitions.append(self._create_transition(
                    from_step=prev_step_id,
                    to_step="output",
                ))
                for s in steps:
                    if s.id == prev_step_id:
                        s.next_steps = ["output"]

        return WorkflowDefinition(
            id=workflow_id,
            name=self.name,
            description=f"Sequential pipeline with {len(stage_configs)} stages",
            steps=steps,
            transitions=transitions,
            entry_step=steps[0].id if steps else None,
            category=self.config.get("category", WorkflowCategory.GENERAL),
            tags=["sequential", "pipeline"] + self.config.get("tags", []),
            metadata={
                "pattern": "sequential",
                "stages": len(stage_configs),
                "pass_full_context": self.pass_full_context,
            },
        )

    def _build_context_prompt(self, prompt: str, prev_step_id: str, focus: str) -> str:
        """Build prompt with context from previous step."""
        context_section = f"""
Previous analysis: {{step.{prev_step_id}}}

"""
        if focus:
            context_section += f"Your focus area: {focus}\n\n"

        return context_section + prompt

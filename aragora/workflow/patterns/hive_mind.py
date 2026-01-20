"""
Hive Mind Pattern - Parallel agent execution with consensus merge.

The Hive Mind pattern executes multiple agents in parallel on the same task,
then merges their responses using a consensus mechanism. This is ideal for:
- Getting diverse perspectives on complex problems
- Reducing individual agent bias
- Increasing reliability through redundancy

Structure:
    [Input] -> [Agent 1] -\
            -> [Agent 2] --> [Consensus Merge] -> [Output]
            -> [Agent 3] -/

Configuration:
    - agents: List of agent types to run in parallel
    - consensus_mode: How to merge results (weighted, majority, synthesis)
    - consensus_threshold: Minimum agreement level (0.0-1.0)
    - include_dissent: Whether to capture dissenting opinions
"""

from __future__ import annotations

from typing import List, Optional

from aragora.workflow.types import (
    WorkflowDefinition,
    Position,
    NodeCategory,
    WorkflowCategory,
)
from aragora.workflow.patterns.base import WorkflowPattern, PatternType


class HiveMindPattern(WorkflowPattern):
    """
    Parallel agent execution with consensus merge.

    All agents process the same input concurrently, and their outputs
    are merged using the specified consensus mechanism.

    Example:
        workflow = HiveMindPattern.create(
            name="Contract Risk Analysis",
            agents=["claude", "gpt4", "gemini"],
            task="Identify potential risks in this contract: {contract}",
            consensus_mode="synthesis",
            consensus_threshold=0.7,
        )
    """

    pattern_type = PatternType.HIVE_MIND

    def __init__(
        self,
        name: str,
        agents: Optional[List[str]] = None,
        task: str = "",
        consensus_mode: str = "synthesis",  # weighted, majority, synthesis
        consensus_threshold: float = 0.7,
        include_dissent: bool = True,
        timeout_per_agent: float = 120.0,
        **kwargs,
    ):
        super().__init__(name, agents, task, **kwargs)
        self.consensus_mode = consensus_mode
        self.consensus_threshold = consensus_threshold
        self.include_dissent = include_dissent
        self.timeout_per_agent = timeout_per_agent

    def create_workflow(self) -> WorkflowDefinition:
        """Create a hive-mind workflow definition."""
        workflow_id = self._generate_id("hm")
        steps = []
        transitions = []

        # Calculate positions for visual layout
        start_x = 100
        agent_y_start = 100
        agent_spacing = 150
        merge_x = 500

        # Create parallel agent steps
        agent_step_ids = []
        for i, agent_type in enumerate(self.agents):
            step_id = f"agent_{agent_type}_{i}"
            agent_step_ids.append(step_id)

            step = self._create_agent_step(
                step_id=step_id,
                name=f"{agent_type.title()} Analysis",
                agent_type=agent_type,
                prompt=self.task,
                position=Position(x=start_x, y=agent_y_start + i * agent_spacing),
                timeout=self.timeout_per_agent,
            )
            steps.append(step)

        # Create consensus merge step
        merge_y = agent_y_start + (len(self.agents) - 1) * agent_spacing / 2
        merge_step = self._create_task_step(
            step_id="consensus_merge",
            name="Consensus Merge",
            task_type="aggregate",
            config={
                "mode": "merge",
                "inputs": agent_step_ids,
            },
            position=Position(x=merge_x, y=merge_y),
            category=NodeCategory.TASK,
        )
        steps.append(merge_step)

        # Create synthesis step (uses first agent to synthesize)
        synthesis_step = self._create_agent_step(
            step_id="synthesis",
            name="Synthesis",
            agent_type=self.agents[0],
            prompt=self._build_synthesis_prompt(),
            position=Position(x=merge_x + 200, y=merge_y),
        )
        synthesis_step.config["system_prompt"] = (
            "You are synthesizing multiple expert perspectives. "
            "Identify common themes, resolve contradictions, and provide a unified analysis."
        )
        steps.append(synthesis_step)

        # Create transitions
        for agent_id in agent_step_ids:
            transitions.append(
                self._create_transition(
                    from_step=agent_id,
                    to_step="consensus_merge",
                )
            )

        transitions.append(
            self._create_transition(
                from_step="consensus_merge",
                to_step="synthesis",
            )
        )

        # Set next_steps for sequential fallback
        for i, step in enumerate(steps[:-1]):
            if step.id in agent_step_ids:
                step.next_steps = ["consensus_merge"]

        merge_step.next_steps = ["synthesis"]

        return WorkflowDefinition(
            id=workflow_id,
            name=self.name,
            description=f"Hive Mind pattern with {len(self.agents)} agents",
            steps=steps,
            transitions=transitions,
            entry_step=agent_step_ids[0] if agent_step_ids else None,
            category=self.config.get("category", WorkflowCategory.GENERAL),
            tags=["hive_mind", "parallel", "consensus"] + self.config.get("tags", []),
            metadata={
                "pattern": "hive_mind",
                "agents": self.agents,
                "consensus_mode": self.consensus_mode,
                "consensus_threshold": self.consensus_threshold,
            },
        )

    def _build_synthesis_prompt(self) -> str:
        """Build the synthesis prompt template."""
        return """Synthesize these expert perspectives on the task.

Task: {task}

Perspectives:
{step.consensus_merge}

Instructions:
1. Identify common themes and agreements
2. Note any significant disagreements or different approaches
3. Provide a unified, comprehensive analysis
4. Highlight the confidence level based on agreement

Provide your synthesis:"""

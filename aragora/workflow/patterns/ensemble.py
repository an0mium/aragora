"""
Ensemble Pattern - Heterogeneous agent pool with best-response selection.

This pattern executes a pool of heterogeneous agents in parallel within a
single agent step (using AgentStep's agent_pool support) and selects the
best response based on a scoring strategy. It is optimized for fast
multi-agent inference without a full consensus merge step.
"""

from __future__ import annotations

from typing import Any

from aragora.workflow.types import WorkflowDefinition, Position, WorkflowCategory
from aragora.workflow.patterns.base import WorkflowPattern, PatternType


class EnsemblePattern(WorkflowPattern):
    """Heterogeneous agent pool with best-response selection."""

    pattern_type = PatternType.ENSEMBLE

    def __init__(
        self,
        name: str,
        agents: list[str] | None = None,
        task: str = "",
        selection_strategy: str = "best_score",
        samples_per_agent: int = 1,
        power_sampling: dict[str, Any] | bool | None = None,
        include_candidates: bool = False,
        max_parallel_agents: int | None = None,
        timeout_seconds: float = 120.0,
        **kwargs,
    ) -> None:
        super().__init__(name, agents, task, **kwargs)
        self.selection_strategy = selection_strategy
        self.samples_per_agent = max(1, int(samples_per_agent))
        self.power_sampling = power_sampling
        self.include_candidates = include_candidates
        self.max_parallel_agents = max_parallel_agents
        self.timeout_seconds = timeout_seconds

    def create_workflow(self) -> WorkflowDefinition:
        """Create an ensemble workflow definition."""
        workflow_id = self._generate_id("ens")
        primary_agent = self.agents[0] if self.agents else "claude"

        step = self._create_agent_step(
            step_id="ensemble_select",
            name="Ensemble Selection",
            agent_type=primary_agent,
            prompt=self.task,
            position=Position(x=120, y=120),
            timeout=self.timeout_seconds,
        )

        step.config.update(
            {
                "agent_pool": self.agents,
                "selection_strategy": self.selection_strategy,
                "samples_per_agent": self.samples_per_agent,
                "include_candidates": self.include_candidates,
            }
        )

        if self.power_sampling is not None:
            step.config["power_sampling"] = self.power_sampling
        if self.max_parallel_agents is not None:
            step.config["max_parallel_agents"] = self.max_parallel_agents

        return WorkflowDefinition(
            id=workflow_id,
            name=self.name,
            description=f"Ensemble pattern with {len(self.agents)} agents",
            steps=[step],
            entry_step=step.id,
            category=self.config.get("category", WorkflowCategory.GENERAL),
            tags=["ensemble", "heterogeneous", "selection"] + self.config.get("tags", []),
            metadata={
                "pattern": "ensemble",
                "agents": self.agents,
                "selection_strategy": self.selection_strategy,
                "samples_per_agent": self.samples_per_agent,
            },
        )


__all__ = ["EnsemblePattern"]

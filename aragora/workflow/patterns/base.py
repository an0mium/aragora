"""
Base classes for workflow patterns.

Provides the foundation for defining reusable workflow patterns that can be
instantiated and customized for specific use cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from aragora.workflow.types import (
    WorkflowDefinition,
    StepDefinition,
    TransitionRule,
    ExecutionPattern,
    VisualNodeData,
    VisualEdgeData,
    Position,
    NodeCategory,
    EdgeType,
    WorkflowCategory,
)


class PatternType(Enum):
    """Types of workflow patterns."""

    HIVE_MIND = "hive_mind"
    SEQUENTIAL = "sequential"
    MAP_REDUCE = "map_reduce"
    HIERARCHICAL = "hierarchical"
    REVIEW_CYCLE = "review_cycle"
    DIALECTIC = "dialectic"
    DEBATE = "debate"
    CUSTOM = "custom"


@dataclass
class ResourceLimits:
    """Resource limits for workflow execution."""

    max_tokens: int = 100000
    max_cost_usd: float = 10.0
    timeout_seconds: float = 600.0
    max_parallel_agents: int = 5
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "timeout_seconds": self.timeout_seconds,
            "max_parallel_agents": self.max_parallel_agents,
            "max_retries": self.max_retries,
        }


@dataclass
class PatternConfig:
    """Configuration for a workflow pattern."""

    name: str
    description: str = ""
    agents: List[str] = field(default_factory=lambda: ["claude", "gpt4"])
    task: str = ""
    category: WorkflowCategory = WorkflowCategory.GENERAL
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Agent configuration
    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Output configuration
    output_format: str = "json"  # json, text, structured


class WorkflowPattern(ABC):
    """
    Abstract base class for workflow patterns.

    Workflow patterns encapsulate common multi-agent orchestration strategies
    as reusable templates. Each pattern defines:
    - How agents are arranged and connected
    - How data flows between steps
    - How results are aggregated
    - Visual layout for the workflow builder

    Subclasses implement specific patterns like HiveMind (parallel + consensus),
    MapReduce (split + map + reduce), Sequential (pipeline), etc.
    """

    pattern_type: PatternType = PatternType.CUSTOM

    def __init__(
        self,
        name: str,
        agents: Optional[List[str]] = None,
        task: str = "",
        **kwargs,
    ):
        self.name = name
        self.agents = agents or ["claude", "gpt4"]
        self.task = task
        self.config = kwargs

    @abstractmethod
    def create_workflow(self) -> WorkflowDefinition:
        """
        Create a WorkflowDefinition from this pattern.

        Returns:
            WorkflowDefinition ready for execution
        """
        ...

    @classmethod
    def create(cls, **kwargs) -> WorkflowDefinition:
        """
        Factory method to create a workflow from pattern configuration.

        Args:
            **kwargs: Pattern-specific configuration

        Returns:
            WorkflowDefinition instance
        """
        pattern = cls(**kwargs)
        return pattern.create_workflow()

    def _generate_id(self, prefix: str = "wf") -> str:
        """Generate a unique workflow/step ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _create_agent_step(
        self,
        step_id: str,
        name: str,
        agent_type: str,
        prompt: str,
        position: Position,
        timeout: float = 120.0,
        retries: int = 1,
    ) -> StepDefinition:
        """Create an agent step definition."""
        return StepDefinition(
            id=step_id,
            name=name,
            step_type="agent",
            config={
                "agent_type": agent_type,
                "prompt_template": prompt,
            },
            timeout_seconds=timeout,
            retries=retries,
            visual=VisualNodeData(
                position=position,
                category=NodeCategory.AGENT,
                color=self._get_agent_color(agent_type),
            ),
        )

    def _create_task_step(
        self,
        step_id: str,
        name: str,
        task_type: str,
        config: Dict[str, Any],
        position: Position,
        category: NodeCategory = NodeCategory.TASK,
    ) -> StepDefinition:
        """Create a task step definition."""
        return StepDefinition(
            id=step_id,
            name=name,
            step_type="task",
            config={"task_type": task_type, **config},
            visual=VisualNodeData(
                position=position,
                category=category,
                color=self._get_category_color(category),
            ),
        )

    def _create_debate_step(
        self,
        step_id: str,
        name: str,
        topic: str,
        agents: List[str],
        position: Position,
        rounds: int = 3,
        consensus_mechanism: str = "majority",
    ) -> StepDefinition:
        """Create a debate step definition."""
        return StepDefinition(
            id=step_id,
            name=name,
            step_type="debate",
            config={
                "topic": topic,
                "agents": agents,
                "rounds": rounds,
                "consensus_mechanism": consensus_mechanism,
            },
            visual=VisualNodeData(
                position=position,
                category=NodeCategory.DEBATE,
                color="#38b2ac",  # Teal
            ),
        )

    def _create_transition(
        self,
        from_step: str,
        to_step: str,
        condition: str = "True",
        label: str = "",
        edge_type: EdgeType = EdgeType.DATA_FLOW,
    ) -> TransitionRule:
        """Create a transition rule between steps."""
        return TransitionRule(
            id=f"tr_{uuid.uuid4().hex[:8]}",
            from_step=from_step,
            to_step=to_step,
            condition=condition,
            label=label,
            visual=VisualEdgeData(
                edge_type=edge_type,
                label=label,
            ),
        )

    def _get_agent_color(self, agent_type: str) -> str:
        """Get color for agent type."""
        colors = {
            "claude": "#7c3aed",  # Purple
            "gpt4": "#10b981",  # Green
            "gpt-4": "#10b981",
            "gemini": "#3b82f6",  # Blue
            "mistral": "#f59e0b",  # Amber
            "grok": "#ef4444",  # Red
            "deepseek": "#06b6d4",  # Cyan
            "llama": "#8b5cf6",  # Violet
        }
        return colors.get(agent_type.lower(), "#6b7280")  # Gray default

    def _get_category_color(self, category: NodeCategory) -> str:
        """Get color for node category."""
        colors = {
            NodeCategory.AGENT: "#4299e1",
            NodeCategory.TASK: "#48bb78",
            NodeCategory.CONTROL: "#ed8936",
            NodeCategory.MEMORY: "#9f7aea",
            NodeCategory.HUMAN: "#f56565",
            NodeCategory.DEBATE: "#38b2ac",
            NodeCategory.INTEGRATION: "#667eea",
        }
        return colors.get(category, "#6b7280")

"""
Hierarchical Pattern - Manager-worker delegation pattern.

The Hierarchical pattern has a manager agent that decomposes tasks
and delegates to specialist worker agents. This is ideal for:
- Complex research tasks
- Multi-domain problems
- Tasks requiring task decomposition

Structure:
    [Input] -> [Manager] -> [Decompose] -> [Worker 1] -\
                                        -> [Worker 2] --> [Manager Review] -> [Output]
                                        -> [Worker N] -/

Configuration:
    - manager_agent: Agent for task decomposition and review
    - worker_agents: Specialist agents for subtasks
    - max_subtasks: Maximum number of subtasks to create
    - delegation_prompt: How manager should decompose tasks
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aragora.workflow.types import (
    WorkflowDefinition,
    StepDefinition,
    Position,
    NodeCategory,
    WorkflowCategory,
    VisualNodeData,
)
from aragora.workflow.patterns.base import WorkflowPattern, PatternType


class HierarchicalPattern(WorkflowPattern):
    """
    Manager-worker delegation pattern.

    A manager agent decomposes the task, delegates subtasks to workers,
    and reviews/integrates the results.

    Example:
        workflow = HierarchicalPattern.create(
            name="Research Project",
            manager_agent="claude",
            worker_agents=["gpt4", "gemini", "claude"],
            task="Research the impact of AI on healthcare",
            max_subtasks=4,
        )
    """

    pattern_type = PatternType.HIERARCHICAL

    def __init__(
        self,
        name: str,
        agents: Optional[List[str]] = None,
        task: str = "",
        manager_agent: Optional[str] = None,
        worker_agents: Optional[List[str]] = None,
        max_subtasks: int = 4,
        delegation_prompt: str = "",
        review_prompt: str = "",
        timeout_per_worker: float = 120.0,
        **kwargs,
    ):
        super().__init__(name, agents, task, **kwargs)
        self.manager_agent = manager_agent or (agents[0] if agents else "claude")
        self.worker_agents = worker_agents or agents or ["gpt4", "gemini"]
        self.max_subtasks = max_subtasks
        self.delegation_prompt = delegation_prompt
        self.review_prompt = review_prompt
        self.timeout_per_worker = timeout_per_worker

    def create_workflow(self) -> WorkflowDefinition:
        """Create a hierarchical workflow definition."""
        workflow_id = self._generate_id("hier")
        steps = []
        transitions = []

        # Calculate positions
        manager_x = 100
        decompose_x = 300
        worker_x = 500
        review_x = 700
        y_base = 200
        worker_spacing = 120

        # Step 1: Manager decomposition
        decompose_prompt = self.delegation_prompt or self._build_decompose_prompt()
        decompose_step = self._create_agent_step(
            step_id="decompose",
            name="Task Decomposition",
            agent_type=self.manager_agent,
            prompt=decompose_prompt,
            position=Position(x=manager_x, y=y_base),
        )
        decompose_step.config["system_prompt"] = (
            "You are a project manager. Decompose complex tasks into clear subtasks. "
            f"Create at most {self.max_subtasks} subtasks. Format as JSON array."
        )
        steps.append(decompose_step)

        # Step 2: Parse subtasks
        parse_step = self._create_task_step(
            step_id="parse_subtasks",
            name="Parse Subtasks",
            task_type="function",
            config={
                "handler": "hierarchical_parse_subtasks",
                "args": {"max_subtasks": self.max_subtasks},
            },
            position=Position(x=decompose_x, y=y_base),
            category=NodeCategory.CONTROL,
        )
        steps.append(parse_step)

        # Step 3: Dispatch to workers (parallel execution)
        dispatch_step = self._create_task_step(
            step_id="dispatch_workers",
            name="Dispatch to Workers",
            task_type="function",
            config={
                "handler": "hierarchical_dispatch",
                "args": {
                    "worker_agents": self.worker_agents,
                    "timeout": self.timeout_per_worker,
                },
            },
            position=Position(x=worker_x, y=y_base),
            category=NodeCategory.AGENT,
        )
        steps.append(dispatch_step)

        # Step 4: Manager review and integration
        review_prompt = self.review_prompt or self._build_review_prompt()
        review_step = self._create_agent_step(
            step_id="review",
            name="Manager Review",
            agent_type=self.manager_agent,
            prompt=review_prompt,
            position=Position(x=review_x, y=y_base),
        )
        review_step.config["system_prompt"] = (
            "You are reviewing work from your team. Integrate findings, "
            "identify gaps, and provide a comprehensive final answer."
        )
        steps.append(review_step)

        # Set up transitions
        decompose_step.next_steps = ["parse_subtasks"]
        parse_step.next_steps = ["dispatch_workers"]
        dispatch_step.next_steps = ["review"]

        transitions.extend([
            self._create_transition("decompose", "parse_subtasks"),
            self._create_transition("parse_subtasks", "dispatch_workers"),
            self._create_transition("dispatch_workers", "review"),
        ])

        return WorkflowDefinition(
            id=workflow_id,
            name=self.name,
            description=f"Hierarchical pattern: {self.manager_agent} manages {len(self.worker_agents)} workers",
            steps=steps,
            transitions=transitions,
            entry_step="decompose",
            category=self.config.get("category", WorkflowCategory.GENERAL),
            tags=["hierarchical", "delegation", "manager-worker"] + self.config.get("tags", []),
            metadata={
                "pattern": "hierarchical",
                "manager_agent": self.manager_agent,
                "worker_agents": self.worker_agents,
                "max_subtasks": self.max_subtasks,
            },
        )

    def _build_decompose_prompt(self) -> str:
        """Build the task decomposition prompt."""
        return f"""Analyze this task and decompose it into {self.max_subtasks} or fewer subtasks.

Task: {{task}}

Instructions:
1. Break down the main task into distinct, actionable subtasks
2. Each subtask should be specific and completable independently
3. Assign a brief title and description to each subtask
4. Order subtasks by logical sequence or priority

Return as JSON array:
[
  {{"title": "Subtask 1", "description": "...", "focus": "area"}},
  {{"title": "Subtask 2", "description": "...", "focus": "area"}}
]

Subtasks:"""

    def _build_review_prompt(self) -> str:
        """Build the manager review prompt."""
        return """Review and integrate the work from your team.

Original Task: {task}

Subtask Results:
{step.dispatch_workers}

Instructions:
1. Review each subtask result for quality and completeness
2. Identify any gaps or contradictions
3. Integrate findings into a coherent response
4. Provide a comprehensive final answer

Final Integrated Response:"""


# Register hierarchical handlers
def _register_hierarchical_handlers():
    """Register hierarchical task handlers."""
    try:
        from aragora.workflow.nodes.task import register_task_handler
        import json

        async def hierarchical_parse_subtasks(context, max_subtasks=4):
            """Parse subtasks from manager decomposition."""
            decompose_result = context.step_outputs.get("decompose", {})
            response = decompose_result.get("response", "")

            # Try to parse JSON from response
            subtasks = []
            try:
                # Find JSON array in response
                import re
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    subtasks = json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError):
                # Fallback: treat response as single task
                subtasks = [{"title": "Main Task", "description": response}]

            # Limit to max_subtasks
            subtasks = subtasks[:max_subtasks]

            return {"subtasks": subtasks, "count": len(subtasks)}

        async def hierarchical_dispatch(context, worker_agents=None, timeout=120.0):
            """Dispatch subtasks to worker agents."""
            import asyncio
            from aragora.agents import create_agent

            worker_agents = worker_agents or ["claude", "gpt4"]
            parse_result = context.step_outputs.get("parse_subtasks", {})
            subtasks = parse_result.get("subtasks", [])
            original_task = context.inputs.get("task", "")

            if not subtasks:
                return {"results": [], "error": "No subtasks to process"}

            async def process_subtask(subtask, index):
                # Round-robin assignment to workers
                agent_type = worker_agents[index % len(worker_agents)]
                try:
                    agent = create_agent(agent_type)
                    prompt = f"""Complete this subtask as part of a larger project.

Original Task: {original_task}

Your Subtask: {subtask.get('title', 'Task')}
Description: {subtask.get('description', '')}
Focus Area: {subtask.get('focus', 'general')}

Provide a thorough response:"""

                    result = await asyncio.wait_for(
                        agent.generate(prompt),
                        timeout=timeout,
                    )
                    return {
                        "subtask": subtask.get("title", f"Subtask {index}"),
                        "agent": agent_type,
                        "result": result,
                        "success": True,
                    }
                except Exception as e:
                    return {
                        "subtask": subtask.get("title", f"Subtask {index}"),
                        "agent": agent_type,
                        "error": str(e),
                        "success": False,
                    }

            tasks = [process_subtask(st, i) for i, st in enumerate(subtasks)]
            results = await asyncio.gather(*tasks)

            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            # Format results for review
            formatted = "\n\n".join([
                f"### {r['subtask']} ({r['agent']})\n{r['result']}"
                for r in successful
            ])

            return {
                "results": successful,
                "failed": failed,
                "formatted": formatted,
                "total": len(subtasks),
            }

        register_task_handler("hierarchical_parse_subtasks", hierarchical_parse_subtasks)
        register_task_handler("hierarchical_dispatch", hierarchical_dispatch)

    except ImportError:
        pass


_register_hierarchical_handlers()

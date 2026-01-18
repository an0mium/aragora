"""
MapReduce Pattern - Split work, parallel processing, aggregate results.

The MapReduce pattern splits input data, processes chunks in parallel,
and aggregates the results. This is ideal for:
- Large document analysis
- Repository/codebase scanning
- Batch processing with aggregation

Structure:
    [Input] -> [Split] -> [Map Agent 1] -\
                       -> [Map Agent 2] --> [Reduce/Aggregate] -> [Output]
                       -> [Map Agent N] -/

Configuration:
    - split_strategy: How to split input (chunks, lines, sections, files)
    - chunk_size: Size of each chunk (for chunk strategy)
    - map_agent: Agent type for map phase
    - reduce_agent: Agent type for reduce phase (optional)
    - parallel_limit: Max concurrent map operations
"""

from __future__ import annotations

from typing import List, Optional

from aragora.workflow.types import (
    WorkflowDefinition,
    StepDefinition,
    Position,
    NodeCategory,
    WorkflowCategory,
)
from aragora.workflow.patterns.base import WorkflowPattern, PatternType


class MapReducePattern(WorkflowPattern):
    """
    Split work, parallel processing, aggregate results.

    Input is split into chunks, each chunk is processed in parallel
    by map agents, and results are aggregated by a reduce step.

    Example:
        workflow = MapReducePattern.create(
            name="Repository Security Scan",
            split_strategy="files",
            file_pattern="**/*.py",
            map_agent="claude",
            map_prompt="Analyze this file for security issues: {chunk}",
            reduce_agent="gpt4",
            reduce_prompt="Aggregate security findings: {map_results}",
        )
    """

    pattern_type = PatternType.MAP_REDUCE

    def __init__(
        self,
        name: str,
        agents: Optional[List[str]] = None,
        task: str = "",
        split_strategy: str = "chunks",  # chunks, lines, sections, files
        chunk_size: int = 4000,
        map_agent: Optional[str] = None,
        map_prompt: str = "",
        reduce_agent: Optional[str] = None,
        reduce_prompt: str = "",
        parallel_limit: int = 5,
        file_pattern: str = "**/*",
        timeout_per_chunk: float = 60.0,
        **kwargs,
    ):
        super().__init__(name, agents, task, **kwargs)
        self.split_strategy = split_strategy
        self.chunk_size = chunk_size
        self.map_agent = map_agent or (agents[0] if agents else "claude")
        self.map_prompt = map_prompt or task
        self.reduce_agent = reduce_agent or (agents[-1] if agents else "gpt4")
        self.reduce_prompt = reduce_prompt
        self.parallel_limit = parallel_limit
        self.file_pattern = file_pattern
        self.timeout_per_chunk = timeout_per_chunk

    def create_workflow(self) -> WorkflowDefinition:
        """Create a map-reduce workflow definition."""
        workflow_id = self._generate_id("mr")
        steps = []
        transitions = []

        # Calculate positions for visual layout
        split_x = 100
        map_x = 350
        reduce_x = 600
        y_pos = 200

        # Step 1: Split step
        split_step = self._create_task_step(
            step_id="split",
            name="Split Input",
            task_type="function",
            config={
                "handler": "map_reduce_split",
                "args": {
                    "strategy": self.split_strategy,
                    "chunk_size": self.chunk_size,
                    "file_pattern": self.file_pattern,
                },
            },
            position=Position(x=split_x, y=y_pos),
            category=NodeCategory.CONTROL,
        )
        steps.append(split_step)

        # Step 2: Map step (conceptually parallel, executed by engine)
        map_step = StepDefinition(
            id="map",
            name="Map (Parallel Processing)",
            step_type="task",
            config={
                "task_type": "function",
                "handler": "map_reduce_map",
                "args": {
                    "agent_type": self.map_agent,
                    "prompt_template": self.map_prompt,
                    "parallel_limit": self.parallel_limit,
                    "timeout_per_chunk": self.timeout_per_chunk,
                },
            },
            visual=VisualNodeData(
                position=Position(x=map_x, y=y_pos),
                category=NodeCategory.AGENT,
                color=self._get_agent_color(self.map_agent),
            ),
        )
        steps.append(map_step)

        # Step 3: Reduce step
        reduce_prompt = self.reduce_prompt or self._build_default_reduce_prompt()
        reduce_step = self._create_agent_step(
            step_id="reduce",
            name="Reduce (Aggregate)",
            agent_type=self.reduce_agent,
            prompt=reduce_prompt,
            position=Position(x=reduce_x, y=y_pos),
        )
        steps.append(reduce_step)

        # Transitions
        split_step.next_steps = ["map"]
        map_step.next_steps = ["reduce"]

        transitions.extend([
            self._create_transition("split", "map"),
            self._create_transition("map", "reduce"),
        ])

        return WorkflowDefinition(
            id=workflow_id,
            name=self.name,
            description=f"MapReduce pattern: {self.split_strategy} split, {self.parallel_limit} parallel workers",
            steps=steps,
            transitions=transitions,
            entry_step="split",
            category=self.config.get("category", WorkflowCategory.GENERAL),
            tags=["map_reduce", "parallel", "batch"] + self.config.get("tags", []),
            metadata={
                "pattern": "map_reduce",
                "split_strategy": self.split_strategy,
                "chunk_size": self.chunk_size,
                "parallel_limit": self.parallel_limit,
                "map_agent": self.map_agent,
                "reduce_agent": self.reduce_agent,
            },
        )

    def _build_default_reduce_prompt(self) -> str:
        """Build default reduce prompt."""
        return """Aggregate and summarize the following analysis results:

Results from parallel processing:
{step.map}

Instructions:
1. Identify common patterns across all results
2. Highlight the most important findings
3. Note any anomalies or unique items
4. Provide a comprehensive summary

Aggregated Analysis:"""


# Register built-in handlers for map-reduce operations
def _register_map_reduce_handlers():
    """Register map-reduce task handlers."""
    try:
        from aragora.workflow.nodes.task import register_task_handler

        async def map_reduce_split(context, strategy="chunks", chunk_size=4000, file_pattern="**/*"):
            """Split input into chunks for parallel processing."""
            input_data = context.inputs.get("input", context.inputs.get("data", ""))

            if strategy == "chunks":
                # Split text into chunks
                if isinstance(input_data, str):
                    chunks = [
                        input_data[i:i + chunk_size]
                        for i in range(0, len(input_data), chunk_size)
                    ]
                else:
                    chunks = [input_data]
            elif strategy == "lines":
                # Split by lines
                if isinstance(input_data, str):
                    chunks = input_data.split("\n")
                else:
                    chunks = [input_data]
            elif strategy == "sections":
                # Split by double newlines (paragraphs/sections)
                if isinstance(input_data, str):
                    chunks = [s.strip() for s in input_data.split("\n\n") if s.strip()]
                else:
                    chunks = [input_data]
            else:
                chunks = [input_data]

            return {"chunks": chunks, "count": len(chunks)}

        async def map_reduce_map(
            context,
            agent_type="claude",
            prompt_template="",
            parallel_limit=5,
            timeout_per_chunk=60.0,
        ):
            """Process chunks in parallel with specified agent."""
            import asyncio
            from aragora.agents import create_agent

            split_result = context.step_outputs.get("split", {})
            chunks = split_result.get("chunks", [])

            if not chunks:
                return {"results": [], "error": "No chunks to process"}

            agent = create_agent(agent_type)
            semaphore = asyncio.Semaphore(parallel_limit)

            async def process_chunk(chunk, index):
                async with semaphore:
                    try:
                        prompt = prompt_template.replace("{chunk}", str(chunk))
                        prompt = prompt.replace("{index}", str(index))
                        result = await asyncio.wait_for(
                            agent.generate(prompt),
                            timeout=timeout_per_chunk,
                        )
                        return {"index": index, "result": result, "success": True}
                    except Exception as e:
                        return {"index": index, "error": str(e), "success": False}

            tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
            results = await asyncio.gather(*tasks)

            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            return {
                "results": successful,
                "failed": failed,
                "total": len(chunks),
                "successful_count": len(successful),
                "failed_count": len(failed),
            }

        register_task_handler("map_reduce_split", map_reduce_split)
        register_task_handler("map_reduce_map", map_reduce_map)

    except ImportError:
        pass


# Import visual types needed
from aragora.workflow.types import VisualNodeData

# Register handlers when module is imported
_register_map_reduce_handlers()

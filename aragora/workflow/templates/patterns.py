"""
Pattern-based Workflow Templates.

Pre-packaged workflow templates based on the core patterns:
- HiveMind: Parallel agent execution with consensus merge
- MapReduce: Split work, parallel processing, aggregate results
- ReviewCycle: Iterative refinement with convergence check

These templates can be customized and used directly or serve as
starting points for domain-specific workflows.

Usage:
    from aragora.workflow.templates.patterns import (
        HIVE_MIND_TEMPLATE,
        MAP_REDUCE_TEMPLATE,
        REVIEW_CYCLE_TEMPLATE,
        create_hive_mind_workflow,
        create_map_reduce_workflow,
        create_review_cycle_workflow,
    )

    # Use pre-built template
    workflow = create_hive_mind_workflow(
        name="Risk Assessment",
        agents=["claude", "gpt4", "gemini"],
        task="Assess risks in this proposal",
    )

    # Or customize the template
    from aragora.workflow.patterns import HiveMindPattern
    custom = HiveMindPattern(
        name="Custom Analysis",
        agents=["claude", "mistral"],
        consensus_mode="weighted",
    ).create_workflow()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from aragora.workflow.patterns import (
    HiveMindPattern,
    MapReducePattern,
    ReviewCyclePattern,
)
from aragora.workflow.templates.package import (
    TemplatePackage,
    TemplateMetadata,
    TemplateAuthor,
    TemplateDependency,
    TemplateStatus,
    TemplateCategory,
    register_package,
)


# ============================================================================
# Hive Mind Template
# ============================================================================

HIVE_MIND_TEMPLATE: Dict[str, Any] = {
    "id": "pattern/hive-mind",
    "name": "Hive Mind Analysis",
    "description": "Parallel multi-agent analysis with consensus synthesis",
    "category": "general",
    "tags": ["parallel", "consensus", "multi-agent", "synthesis"],
    "pattern": "hive_mind",
    "version": "1.0.0",
    "config": {
        "agents": ["claude", "gpt4", "gemini"],
        "consensus_mode": "synthesis",
        "consensus_threshold": 0.7,
        "include_dissent": True,
        "timeout_per_agent": 120.0,
    },
    "inputs": {
        "task": {"type": "string", "required": True, "description": "Task to analyze"},
        "context": {"type": "string", "required": False, "description": "Additional context"},
        "data": {"type": "any", "required": False, "description": "Data to include in analysis"},
    },
    "outputs": {
        "perspectives": {"type": "array", "description": "Individual agent responses"},
        "synthesis": {"type": "string", "description": "Unified analysis"},
        "confidence": {"type": "number", "description": "Confidence level (0-1)"},
        "dissent": {"type": "array", "description": "Dissenting opinions if any"},
    },
}


def create_hive_mind_workflow(
    name: str = "Hive Mind Analysis",
    agents: Optional[List[str]] = None,
    task: str = "",
    consensus_mode: str = "synthesis",
    consensus_threshold: float = 0.7,
    include_dissent: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Create a hive-mind workflow definition.

    Args:
        name: Workflow name
        agents: List of agent types to use
        task: Task prompt template
        consensus_mode: How to merge results (weighted, majority, synthesis)
        consensus_threshold: Minimum agreement level (0.0-1.0)
        include_dissent: Whether to capture dissenting opinions
        **kwargs: Additional pattern configuration

    Returns:
        WorkflowDefinition instance
    """
    pattern = HiveMindPattern(
        name=name,
        agents=agents or ["claude", "gpt4", "gemini"],
        task=task,
        consensus_mode=consensus_mode,
        consensus_threshold=consensus_threshold,
        include_dissent=include_dissent,
        **kwargs,
    )
    return cast(dict[str, Any], pattern.create_workflow())


# ============================================================================
# MapReduce Template
# ============================================================================

MAP_REDUCE_TEMPLATE: Dict[str, Any] = {
    "id": "pattern/map-reduce",
    "name": "MapReduce Processing",
    "description": "Split large inputs, process in parallel, aggregate results",
    "category": "general",
    "tags": ["parallel", "batch", "map-reduce", "scalable"],
    "pattern": "map_reduce",
    "version": "1.0.0",
    "config": {
        "split_strategy": "chunks",
        "chunk_size": 4000,
        "map_agent": "claude",
        "reduce_agent": "gpt4",
        "parallel_limit": 5,
        "timeout_per_chunk": 60.0,
    },
    "inputs": {
        "input": {"type": "string", "required": True, "description": "Text/data to split and process"},
        "task": {"type": "string", "required": True, "description": "Processing task for each chunk"},
        "split_strategy": {"type": "string", "required": False, "description": "How to split (chunks, lines, sections)"},
        "chunk_size": {"type": "integer", "required": False, "description": "Size of chunks"},
    },
    "outputs": {
        "chunks": {"type": "array", "description": "Split input chunks"},
        "map_results": {"type": "array", "description": "Results from parallel processing"},
        "aggregated": {"type": "string", "description": "Final aggregated analysis"},
        "statistics": {"type": "object", "description": "Processing statistics"},
    },
}


def create_map_reduce_workflow(
    name: str = "MapReduce Processing",
    split_strategy: str = "chunks",
    chunk_size: int = 4000,
    map_agent: str = "claude",
    map_prompt: str = "",
    reduce_agent: str = "gpt4",
    reduce_prompt: str = "",
    parallel_limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Create a map-reduce workflow definition.

    Args:
        name: Workflow name
        split_strategy: How to split input (chunks, lines, sections, files)
        chunk_size: Size of each chunk (for chunk strategy)
        map_agent: Agent type for map phase
        map_prompt: Prompt template for map phase
        reduce_agent: Agent type for reduce phase
        reduce_prompt: Prompt template for reduce phase
        parallel_limit: Max concurrent map operations
        **kwargs: Additional pattern configuration

    Returns:
        WorkflowDefinition instance
    """
    pattern = MapReducePattern(
        name=name,
        split_strategy=split_strategy,
        chunk_size=chunk_size,
        map_agent=map_agent,
        map_prompt=map_prompt,
        reduce_agent=reduce_agent,
        reduce_prompt=reduce_prompt,
        parallel_limit=parallel_limit,
        **kwargs,
    )
    return cast(dict[str, Any], pattern.create_workflow())


# ============================================================================
# Review Cycle Template
# ============================================================================

REVIEW_CYCLE_TEMPLATE: Dict[str, Any] = {
    "id": "pattern/review-cycle",
    "name": "Iterative Review Cycle",
    "description": "Iterative refinement with quality threshold convergence",
    "category": "general",
    "tags": ["iterative", "refinement", "review", "quality"],
    "pattern": "review_cycle",
    "version": "1.0.0",
    "config": {
        "draft_agent": "claude",
        "review_agent": "gpt4",
        "max_iterations": 3,
        "convergence_threshold": 0.85,
        "review_criteria": ["quality", "completeness", "accuracy"],
        "timeout_per_step": 120.0,
    },
    "inputs": {
        "task": {"type": "string", "required": True, "description": "What to create/accomplish"},
        "context": {"type": "string", "required": False, "description": "Additional context"},
        "criteria": {"type": "array", "required": False, "description": "Review criteria"},
        "max_iterations": {"type": "integer", "required": False, "description": "Maximum refinement cycles"},
        "threshold": {"type": "number", "required": False, "description": "Convergence threshold (0-1)"},
    },
    "outputs": {
        "final_output": {"type": "string", "description": "The refined result"},
        "iterations": {"type": "integer", "description": "Number of iterations used"},
        "final_score": {"type": "number", "description": "Final quality score"},
        "review_history": {"type": "array", "description": "History of reviews"},
    },
}


def create_review_cycle_workflow(
    name: str = "Iterative Review Cycle",
    draft_agent: str = "claude",
    review_agent: str = "gpt4",
    task: str = "",
    max_iterations: int = 3,
    convergence_threshold: float = 0.85,
    review_criteria: Optional[List[str]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Create a review cycle workflow definition.

    Args:
        name: Workflow name
        draft_agent: Agent for creating/refining drafts
        review_agent: Agent for reviewing
        task: Task prompt template
        max_iterations: Maximum refinement cycles
        convergence_threshold: Score needed to exit loop (0.0-1.0)
        review_criteria: What aspects to review
        **kwargs: Additional pattern configuration

    Returns:
        WorkflowDefinition instance
    """
    pattern = ReviewCyclePattern(
        name=name,
        draft_agent=draft_agent,
        review_agent=review_agent,
        task=task,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        review_criteria=review_criteria or ["quality", "completeness", "accuracy"],
        **kwargs,
    )
    return cast(dict[str, Any], pattern.create_workflow())


# ============================================================================
# Pattern Template Registry
# ============================================================================

PATTERN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "pattern/hive-mind": HIVE_MIND_TEMPLATE,
    "pattern/map-reduce": MAP_REDUCE_TEMPLATE,
    "pattern/review-cycle": REVIEW_CYCLE_TEMPLATE,
}


def get_pattern_template(pattern_id: str) -> Optional[Dict[str, Any]]:
    """Get a pattern template by ID."""
    return PATTERN_TEMPLATES.get(pattern_id)


def list_pattern_templates() -> List[Dict[str, Any]]:
    """List all available pattern templates."""
    templates = []
    for template_id, template in PATTERN_TEMPLATES.items():
        templates.append({
            "id": template_id,
            "name": template.get("name", template_id),
            "description": template.get("description", ""),
            "pattern": template.get("pattern", ""),
            "category": "pattern",
        })
    return templates


# ============================================================================
# Package Registration
# ============================================================================

def register_pattern_packages() -> None:
    """Register all pattern templates as packages."""
    author = TemplateAuthor(
        name="Aragora Team",
        organization="Aragora",
        email="team@aragora.ai",
    )

    # Hive Mind Package
    hive_mind_pkg = TemplatePackage(
        metadata=TemplateMetadata(
            id="pattern/hive-mind",
            name="Hive Mind Analysis",
            version="1.0.0",
            description="Parallel multi-agent analysis with consensus synthesis",
            long_description="""
# Hive Mind Pattern

Execute multiple agents in parallel on the same task, then merge their
responses using a consensus mechanism.

## Use Cases
- Getting diverse perspectives on complex problems
- Reducing individual agent bias
- Increasing reliability through redundancy
- Expert panel simulations

## Configuration
- **agents**: List of agent types to run in parallel
- **consensus_mode**: How to merge results (weighted, majority, synthesis)
- **consensus_threshold**: Minimum agreement level (0.0-1.0)
- **include_dissent**: Whether to capture dissenting opinions
""",
            category=TemplateCategory.GENERAL,
            tags=["parallel", "consensus", "multi-agent", "synthesis"],
            status=TemplateStatus.STABLE,
            author=author,
            estimated_duration="2-5 minutes",
            complexity="medium",
            recommended_agents=["claude", "gpt4", "gemini"],
            dependencies=[
                TemplateDependency(name="claude", type="agent"),
                TemplateDependency(name="gpt4", type="agent", required=False),
                TemplateDependency(name="gemini", type="agent", required=False),
            ],
        ),
        template=HIVE_MIND_TEMPLATE,
        readme="""
# Hive Mind Analysis Template

## Quick Start

```python
from aragora.workflow.templates.patterns import create_hive_mind_workflow

workflow = create_hive_mind_workflow(
    name="Risk Assessment",
    agents=["claude", "gpt4", "gemini"],
    task="Assess risks in this proposal: {proposal}",
)

engine = WorkflowEngine()
result = await engine.execute(workflow, {"proposal": "..."})
```

## Examples

### Contract Analysis
```python
workflow = create_hive_mind_workflow(
    name="Contract Risk Analysis",
    agents=["claude", "gpt4"],
    task="Identify risks in this contract",
    consensus_mode="weighted",
)
```

### Research Synthesis
```python
workflow = create_hive_mind_workflow(
    name="Research Synthesis",
    agents=["claude", "gemini", "mistral"],
    task="Summarize findings from these papers",
    include_dissent=True,
)
```
""",
        examples=[
            {
                "name": "Basic Analysis",
                "inputs": {"task": "Analyze this business proposal", "data": "..."},
                "config": {"agents": ["claude", "gpt4"]},
            },
        ],
    )
    register_package(hive_mind_pkg)

    # MapReduce Package
    map_reduce_pkg = TemplatePackage(
        metadata=TemplateMetadata(
            id="pattern/map-reduce",
            name="MapReduce Processing",
            version="1.0.0",
            description="Split large inputs, process in parallel, aggregate results",
            long_description="""
# MapReduce Pattern

Split input data, process chunks in parallel, and aggregate the results.

## Use Cases
- Large document analysis
- Repository/codebase scanning
- Batch processing with aggregation
- Log analysis

## Configuration
- **split_strategy**: How to split input (chunks, lines, sections, files)
- **chunk_size**: Size of each chunk
- **map_agent**: Agent for processing chunks
- **reduce_agent**: Agent for aggregation
- **parallel_limit**: Max concurrent operations
""",
            category=TemplateCategory.GENERAL,
            tags=["parallel", "batch", "map-reduce", "scalable"],
            status=TemplateStatus.STABLE,
            author=author,
            estimated_duration="3-10 minutes",
            complexity="medium",
            recommended_agents=["claude", "gpt4"],
            dependencies=[
                TemplateDependency(name="claude", type="agent"),
                TemplateDependency(name="map_reduce_split", type="step_type"),
                TemplateDependency(name="map_reduce_map", type="step_type"),
            ],
        ),
        template=MAP_REDUCE_TEMPLATE,
        readme="""
# MapReduce Processing Template

## Quick Start

```python
from aragora.workflow.templates.patterns import create_map_reduce_workflow

workflow = create_map_reduce_workflow(
    name="Document Analysis",
    split_strategy="chunks",
    chunk_size=4000,
    map_prompt="Analyze this section: {chunk}",
)

engine = WorkflowEngine()
result = await engine.execute(workflow, {"input": large_document})
```

## Split Strategies

- **chunks**: Split text into fixed-size chunks
- **lines**: Split by newlines
- **sections**: Split by paragraph/section boundaries
- **files**: Split by files in a directory
""",
        examples=[
            {
                "name": "Document Analysis",
                "inputs": {"input": "...", "task": "Summarize key points"},
                "config": {"split_strategy": "chunks", "chunk_size": 4000},
            },
        ],
    )
    register_package(map_reduce_pkg)

    # Review Cycle Package
    review_cycle_pkg = TemplatePackage(
        metadata=TemplateMetadata(
            id="pattern/review-cycle",
            name="Iterative Review Cycle",
            version="1.0.0",
            description="Iterative refinement with quality threshold convergence",
            long_description="""
# Review Cycle Pattern

Iteratively refine output through multiple rounds of review until
convergence or a maximum iteration count.

## Use Cases
- Code review workflows
- Document editing and refinement
- Quality assurance processes
- Iterative improvement tasks

## Configuration
- **draft_agent**: Agent for creating/refining drafts
- **review_agent**: Agent for reviewing
- **max_iterations**: Maximum refinement cycles
- **convergence_threshold**: Score needed to exit loop
- **review_criteria**: What aspects to review
""",
            category=TemplateCategory.GENERAL,
            tags=["iterative", "refinement", "review", "quality"],
            status=TemplateStatus.STABLE,
            author=author,
            estimated_duration="5-15 minutes",
            complexity="medium",
            recommended_agents=["claude", "gpt4"],
            dependencies=[
                TemplateDependency(name="claude", type="agent"),
                TemplateDependency(name="gpt4", type="agent", required=False),
                TemplateDependency(name="review_cycle_check", type="step_type"),
            ],
        ),
        template=REVIEW_CYCLE_TEMPLATE,
        readme="""
# Iterative Review Cycle Template

## Quick Start

```python
from aragora.workflow.templates.patterns import create_review_cycle_workflow

workflow = create_review_cycle_workflow(
    name="Code Implementation",
    draft_agent="claude",
    review_agent="gpt4",
    task="Implement a rate limiter class",
    review_criteria=["correctness", "efficiency", "readability"],
)

engine = WorkflowEngine()
result = await engine.execute(workflow, {"task": "..."})
```

## Convergence

The cycle continues until:
1. Score >= threshold (convergence)
2. Iterations >= max_iterations

Each iteration:
1. Draft agent creates/refines content
2. Review agent evaluates and scores
3. Feedback is used for next iteration
""",
        examples=[
            {
                "name": "Code Review",
                "inputs": {"task": "Implement a function to validate emails"},
                "config": {"max_iterations": 3, "threshold": 0.85},
            },
        ],
    )
    register_package(review_cycle_pkg)


# Register packages on module import
register_pattern_packages()


__all__ = [
    # Templates
    "HIVE_MIND_TEMPLATE",
    "MAP_REDUCE_TEMPLATE",
    "REVIEW_CYCLE_TEMPLATE",
    "PATTERN_TEMPLATES",
    # Factory functions
    "create_hive_mind_workflow",
    "create_map_reduce_workflow",
    "create_review_cycle_workflow",
    # Registry
    "get_pattern_template",
    "list_pattern_templates",
    "register_pattern_packages",
]

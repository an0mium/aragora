"""
Aragora Pipelines - High-level workflows for complex tasks.

Pipelines combine multiple connectors and processors to accomplish
sophisticated multi-step tasks:

- EssaySynthesisPipeline: Transform conversations into structured essays
- EssayWorkflow: Complete end-to-end essay synthesis with debate
"""

from aragora.pipelines.essay_synthesis import (
    EssaySynthesisPipeline,
    TopicCluster,
    AttributedClaim,
    EssayOutline,
    EssaySection,
    SynthesisConfig,
    create_essay_pipeline,
)
from aragora.pipelines.essay_workflow import (
    EssayWorkflow,
    WorkflowConfig,
    WorkflowResult,
    DebateResult,
    SeedEssay,
    create_workflow,
)

__all__ = [
    # Essay Synthesis
    "EssaySynthesisPipeline",
    "TopicCluster",
    "AttributedClaim",
    "EssayOutline",
    "EssaySection",
    "SynthesisConfig",
    "create_essay_pipeline",
    # Essay Workflow
    "EssayWorkflow",
    "WorkflowConfig",
    "WorkflowResult",
    "DebateResult",
    "SeedEssay",
    "create_workflow",
]

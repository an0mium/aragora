"""
Aragora Pipelines - High-level workflows for complex tasks.

Pipelines combine multiple connectors and processors to accomplish
sophisticated multi-step tasks:

- EssaySynthesisPipeline: Transform conversations into structured essays (claim-based)
- EssayWorkflow: Complete end-to-end essay synthesis with debate
- ProseSynthesisPipeline: Prose-preserving synthesis that maintains original text quality
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
from aragora.pipelines.prose_synthesis import (
    ProseSynthesisPipeline,
    ProsePassage,
    ThemeConfig,
    SynthesisResult,
    create_prose_pipeline,
)

__all__ = [
    # Essay Synthesis (claim-based)
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
    # Prose Synthesis (prose-preserving)
    "ProseSynthesisPipeline",
    "ProsePassage",
    "ThemeConfig",
    "SynthesisResult",
    "create_prose_pipeline",
]

"""
Post-Debate Workflow Pattern - Automated actions after debate completion.

The Post-Debate pattern triggers automated follow-up actions when debates
complete with high confidence. This is ideal for:
- Knowledge extraction and storage
- Documentation generation
- Stakeholder notification
- Triggering dependent workflows

Structure:
    [Debate Result] -> [Extract Knowledge] -> [Store in KM] -> [Notify] -> [Complete]

Configuration:
    - min_confidence: Minimum confidence to trigger (default 0.7)
    - store_consensus: Store consensus in Knowledge Mound
    - notify_webhook: Optional webhook URL for notifications
    - extract_facts: Extract structured facts from debate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from aragora.workflow.types import (
    WorkflowDefinition,
    StepDefinition,
    TransitionRule,
    Position,
    NodeCategory,
    WorkflowCategory,
    VisualNodeData,
    VisualEdgeData,
    EdgeType,
)
from aragora.workflow.patterns.base import WorkflowPattern, PatternType


@dataclass
class PostDebateConfig:
    """Configuration for post-debate workflow."""

    store_consensus: bool = True  # Store consensus in Knowledge Mound
    extract_facts: bool = True  # Extract structured facts
    notify_webhook: Optional[str] = None  # Optional webhook URL
    generate_summary: bool = True  # Generate human-readable summary
    workspace_id: Optional[str] = None  # Target workspace for storage


class PostDebatePattern(WorkflowPattern):
    """
    Post-debate workflow pattern for automated follow-up actions.

    Triggered after debates complete with sufficient confidence to:
    1. Extract knowledge and facts from consensus
    2. Store in Knowledge Mound for future reference
    3. Optionally notify stakeholders
    4. Generate summaries for documentation

    Example:
        workflow = PostDebatePattern.create(
            name="Post-Debate Processing",
            config=PostDebateConfig(
                store_consensus=True,
                extract_facts=True,
            ),
        )
    """

    pattern_type = PatternType.CUSTOM

    def __init__(
        self,
        name: str = "Post-Debate Workflow",
        config: Optional[PostDebateConfig] = None,
        agents: Optional[List[str]] = None,
        task: str = "Process debate outcome",
        **kwargs,
    ):
        super().__init__(name, agents or ["claude"], task, **kwargs)
        self.config = config or PostDebateConfig()

    def create_workflow(self) -> WorkflowDefinition:
        """Create a post-debate workflow definition."""
        workflow_id = f"post_debate_{uuid4().hex[:8]}"
        steps: List[StepDefinition] = []
        transitions: List[TransitionRule] = []

        # Step positions
        y_pos = 200
        step_width = 250

        # Step 1: Extract Knowledge (if enabled)
        current_x = 100
        if self.config.extract_facts:
            extract_step = StepDefinition(
                id="extract_knowledge",
                name="Extract Knowledge",
                step_type="function",
                config={
                    "handler": "extract_debate_facts",
                    "args": {
                        "include_dissent": True,
                        "min_confidence": 0.5,
                    },
                },
                visual=VisualNodeData(
                    position=Position(x=current_x, y=y_pos),
                    category=NodeCategory.TASK,
                ),
            )
            steps.append(extract_step)

            # Transition to next step
            next_step = "store_consensus" if self.config.store_consensus else "generate_summary"
            transitions.append(
                TransitionRule(
                    id=f"tr_{uuid4().hex[:8]}",
                    from_step="extract_knowledge",
                    to_step=next_step,
                    condition="True",  # Always proceed
                    visual=VisualEdgeData(edge_type=EdgeType.DATA_FLOW),
                )
            )
            current_x += step_width

        # Step 2: Store in Knowledge Mound (if enabled)
        if self.config.store_consensus:
            store_step = StepDefinition(
                id="store_consensus",
                name="Store in Knowledge Mound",
                step_type="function",
                config={
                    "handler": "store_debate_consensus",
                    "args": {
                        "workspace_id": self.config.workspace_id or "default",
                        "include_provenance": True,
                    },
                },
                visual=VisualNodeData(
                    position=Position(x=current_x, y=y_pos),
                    category=NodeCategory.MEMORY,
                ),
            )
            steps.append(store_step)

            # Transition to next step
            next_step = "generate_summary" if self.config.generate_summary else "complete"
            transitions.append(
                TransitionRule(
                    id=f"tr_{uuid4().hex[:8]}",
                    from_step="store_consensus",
                    to_step=next_step,
                    condition="True",  # Always proceed
                    visual=VisualEdgeData(edge_type=EdgeType.DATA_FLOW),
                )
            )
            current_x += step_width

        # Step 3: Generate Summary (if enabled)
        if self.config.generate_summary:
            summary_step = StepDefinition(
                id="generate_summary",
                name="Generate Summary",
                step_type="agent",
                config={
                    "agent_type": self.agents[0] if self.agents else "claude",
                    "prompt": (
                        "Summarize the debate outcome in a clear, concise format suitable "
                        "for documentation. Include: key points of agreement, main arguments, "
                        "and confidence level."
                    ),
                    "max_tokens": 500,
                },
                visual=VisualNodeData(
                    position=Position(x=current_x, y=y_pos),
                    category=NodeCategory.AGENT,
                ),
            )
            steps.append(summary_step)

            # Transition to complete or notify
            next_step = "notify" if self.config.notify_webhook else "complete"
            transitions.append(
                TransitionRule(
                    id=f"tr_{uuid4().hex[:8]}",
                    from_step="generate_summary",
                    to_step=next_step,
                    condition="True",  # Always proceed
                    visual=VisualEdgeData(edge_type=EdgeType.DATA_FLOW),
                )
            )
            current_x += step_width

        # Step 4: Notify (if webhook configured)
        if self.config.notify_webhook:
            notify_step = StepDefinition(
                id="notify",
                name="Notify Stakeholders",
                step_type="function",
                config={
                    "handler": "webhook_notify",
                    "args": {
                        "url": self.config.notify_webhook,
                        "include_summary": True,
                    },
                },
                visual=VisualNodeData(
                    position=Position(x=current_x, y=y_pos),
                    category=NodeCategory.INTEGRATION,
                ),
            )
            steps.append(notify_step)
            transitions.append(
                TransitionRule(
                    id=f"tr_{uuid4().hex[:8]}",
                    from_step="notify",
                    to_step="complete",
                    condition="True",  # Always proceed
                    visual=VisualEdgeData(edge_type=EdgeType.DATA_FLOW),
                )
            )
            current_x += step_width

        # Final step: Complete
        complete_step = StepDefinition(
            id="complete",
            name="Complete",
            step_type="transform",
            config={
                "transform": "{'status': 'completed', 'outputs': outputs}",
            },
            visual=VisualNodeData(
                position=Position(x=current_x, y=y_pos),
                category=NodeCategory.INTEGRATION,
            ),
        )
        steps.append(complete_step)

        # Determine entry step
        if self.config.extract_facts:
            entry_step = "extract_knowledge"
        elif self.config.store_consensus:
            entry_step = "store_consensus"
        elif self.config.generate_summary:
            entry_step = "generate_summary"
        else:
            entry_step = "complete"

        return WorkflowDefinition(
            id=workflow_id,
            name=self.name,
            description="Automated post-debate processing workflow",
            steps=steps,
            transitions=transitions,
            entry_step=entry_step,
            inputs={
                "debate_id": "string",
                "consensus": "string",
                "confidence": "float",
                "agents": "list",
            },
            outputs={
                "status": "string",
                "knowledge_stored": "boolean",
                "summary": "string",
            },
            category=WorkflowCategory.GENERAL,
            tags=["post-debate", "automation", "knowledge-extraction"],
        )

    @classmethod
    def create(
        cls,
        name: str = "Post-Debate Workflow",
        store_consensus: bool = True,
        extract_facts: bool = True,
        generate_summary: bool = True,
        notify_webhook: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> WorkflowDefinition:
        """
        Factory method to create a post-debate workflow.

        Args:
            name: Workflow name
            store_consensus: Store consensus in Knowledge Mound
            extract_facts: Extract structured facts from debate
            generate_summary: Generate human-readable summary
            notify_webhook: Optional webhook URL for notifications
            workspace_id: Target workspace for storage

        Returns:
            WorkflowDefinition ready for execution
        """
        config = PostDebateConfig(
            store_consensus=store_consensus,
            extract_facts=extract_facts,
            generate_summary=generate_summary,
            notify_webhook=notify_webhook,
            workspace_id=workspace_id,
        )
        pattern = cls(name=name, config=config, **kwargs)
        return pattern.create_workflow()


def get_default_post_debate_workflow(
    workspace_id: Optional[str] = None,
) -> WorkflowDefinition:
    """
    Get the default post-debate workflow for automatic triggering.

    This workflow is used when enable_post_debate_workflow=True in ArenaConfig
    but no custom workflow is specified.

    Args:
        workspace_id: Optional workspace ID for knowledge storage

    Returns:
        Default WorkflowDefinition for post-debate processing
    """
    return PostDebatePattern.create(
        name="Default Post-Debate Workflow",
        store_consensus=True,
        extract_facts=True,
        generate_summary=True,
        workspace_id=workspace_id,
    )


__all__ = [
    "PostDebatePattern",
    "PostDebateConfig",
    "get_default_post_debate_workflow",
]

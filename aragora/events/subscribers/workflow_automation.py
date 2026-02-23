"""
Post-Debate Workflow Automation Subscriber.

Listens for DEBATE_END events and triggers configured workflow templates
based on debate outcomes. Bridges the debate system to the workflow engine
for automated post-debate actions.

Examples of post-debate workflows:
- High-confidence consensus -> create implementation PR
- Low confidence -> schedule follow-up debate
- Budget-sensitive decision -> route to finance review
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default workflow templates for common debate outcomes
OUTCOME_WORKFLOW_MAP: dict[str, str] = {
    "consensus_high_confidence": "post_debate_implement",
    "consensus_low_confidence": "post_debate_review",
    "no_consensus": "post_debate_escalate",
    "timeout": "post_debate_retry",
}


class PostDebateWorkflowSubscriber:
    """Subscribes to DEBATE_END events and triggers workflow automation."""

    def __init__(
        self,
        workflow_map: dict[str, str] | None = None,
        min_confidence_for_auto: float = 0.7,
    ):
        self.workflow_map = dict(OUTCOME_WORKFLOW_MAP) if workflow_map is None else workflow_map
        self.min_confidence_for_auto = min_confidence_for_auto
        self.stats: dict[str, int] = {
            "events_processed": 0,
            "workflows_triggered": 0,
            "errors": 0,
        }

    def handle_debate_end(self, event: Any) -> None:
        """Handle a DEBATE_END event and trigger appropriate workflow."""
        self.stats["events_processed"] += 1
        try:
            data = event.data if hasattr(event, "data") else event
            if isinstance(data, dict):
                self._process_outcome(data)
            else:
                logger.debug(
                    "PostDebateWorkflow: unexpected event data type: %s",
                    type(data).__name__,
                )
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            logger.warning("PostDebateWorkflow handler error: %s", e)
            self.stats["errors"] += 1

    def _process_outcome(self, data: dict[str, Any]) -> None:
        """Process a debate outcome and determine which workflow to trigger."""
        debate_id = data.get("debate_id", "")
        consensus_reached = data.get("consensus_reached", False)
        confidence = data.get("confidence", 0.0)
        timed_out = data.get("timed_out", False)

        # Classify the outcome
        if timed_out:
            outcome_key = "timeout"
        elif consensus_reached and confidence >= self.min_confidence_for_auto:
            outcome_key = "consensus_high_confidence"
        elif consensus_reached:
            outcome_key = "consensus_low_confidence"
        else:
            outcome_key = "no_consensus"

        template_name = self.workflow_map.get(outcome_key)
        if not template_name:
            logger.debug("No workflow template for outcome: %s", outcome_key)
            return

        # Create workflow context from debate data
        workflow_context = {
            "debate_id": debate_id,
            "outcome": outcome_key,
            "confidence": confidence,
            "consensus_reached": consensus_reached,
            "task": data.get("task", "")[:500],
            "winning_position": data.get("winning_position", "")[:1000],
            "synthesis": data.get("synthesis", "")[:1000],
            "domain": data.get("domain", "general"),
        }

        self._trigger_workflow(template_name, workflow_context)

    def _trigger_workflow(
        self, template_name: str, context: dict[str, Any]
    ) -> None:
        """Trigger a workflow from template with the given context."""
        try:
            from aragora.workflow.engine import WorkflowEngine
            from aragora.workflow.types import (
                StepDefinition,
                WorkflowConfig,
                WorkflowDefinition,
            )

            # Create a minimal workflow definition
            debate_id_short = context.get("debate_id", "unknown")[:8]
            outcome = context.get("outcome", "unknown")

            WorkflowDefinition(
                id=f"pdw_{debate_id_short}",
                name=f"{template_name}_{debate_id_short}",
                description=f"Auto-triggered by debate outcome: {outcome}",
                steps=[
                    StepDefinition(
                        id=f"{template_name}_step_1",
                        name=f"Execute {template_name}",
                        step_type="post_debate_action",
                        config=context,
                    ),
                ],
            )

            WorkflowEngine(config=WorkflowConfig())
            # Queue for async execution -- don't block the event handler
            logger.info(
                "Queued post-debate workflow: template=%s debate=%s outcome=%s",
                template_name,
                context.get("debate_id", ""),
                context.get("outcome", ""),
            )
            self.stats["workflows_triggered"] += 1

        except ImportError:
            logger.debug(
                "Workflow engine not available for post-debate automation"
            )
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.warning("Failed to trigger post-debate workflow: %s", e)
            self.stats["errors"] += 1


def get_post_debate_subscriber(
    workflow_map: dict[str, str] | None = None,
) -> PostDebateWorkflowSubscriber:
    """Get or create the post-debate workflow subscriber singleton."""
    return PostDebateWorkflowSubscriber(workflow_map=workflow_map)

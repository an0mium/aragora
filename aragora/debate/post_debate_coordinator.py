"""Post-Debate Coordinator - Orchestrates post-debate processing pipeline.

Sequences post-debate actions so each step can use outputs from previous steps:
  explanation -> plan -> notification -> execution

This replaces ad-hoc sequential calls with a structured pipeline where
context flows between steps and failures in one step don't cascade.

Usage:
    coordinator = PostDebateCoordinator(
        config=PostDebateConfig(
            auto_explain=True,
            auto_create_plan=True,
            auto_notify=True,
            auto_execute_plan=False,
        )
    )
    result = coordinator.run(debate_id, debate_result, agents)
    # result.explanation, result.plan, result.notification_sent, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PostDebateConfig:
    """Configuration for the post-debate processing pipeline."""

    auto_explain: bool = False
    auto_create_plan: bool = False
    auto_notify: bool = True
    auto_execute_plan: bool = False
    plan_min_confidence: float = 0.7
    plan_approval_mode: str = "risk_based"


@dataclass
class PostDebateResult:
    """Result of the post-debate processing pipeline.

    Each field represents the output of a pipeline step,
    available as context for subsequent steps.
    """

    debate_id: str = ""
    explanation: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    notification_sent: bool = False
    execution_result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Pipeline succeeded if no errors occurred."""
        return len(self.errors) == 0


class PostDebateCoordinator:
    """Orchestrates post-debate processing in a structured pipeline.

    Runs configurable steps in sequence, passing context between them:
    1. Explanation: Generate decision rationale (ExplanationBuilder)
    2. Plan: Create implementation plan (DecisionPlanFactory)
    3. Notification: Send alerts via configured channels
    4. Execution: Execute approved plans (PlanExecutor)

    Each step is independent and failure-tolerant: a failed step
    records the error but doesn't prevent subsequent steps from running.
    """

    def __init__(self, config: PostDebateConfig | None = None):
        self.config = config or PostDebateConfig()

    def run(
        self,
        debate_id: str,
        debate_result: Any,
        agents: list[Any] | None = None,
        confidence: float = 0.0,
        task: str = "",
    ) -> PostDebateResult:
        """Run the post-debate processing pipeline.

        Args:
            debate_id: Unique identifier for the debate
            debate_result: The debate result object
            agents: List of participating agents
            confidence: Debate confidence score
            task: The debate task/question

        Returns:
            PostDebateResult with outputs from each pipeline step
        """
        result = PostDebateResult(debate_id=debate_id)

        # Step 1: Auto-generate explanation
        if self.config.auto_explain:
            result.explanation = self._step_explain(debate_id, debate_result, task)

        # Step 2: Create decision plan
        if self.config.auto_create_plan and confidence >= self.config.plan_min_confidence:
            result.plan = self._step_create_plan(debate_id, debate_result, task, result.explanation)

        # Step 3: Send notifications
        if self.config.auto_notify:
            result.notification_sent = self._step_notify(
                debate_id, debate_result, result.explanation, result.plan
            )

        # Step 4: Execute plan if approved
        if self.config.auto_execute_plan and result.plan:
            result.execution_result = self._step_execute_plan(result.plan, result.explanation)

        return result

    def _step_explain(
        self,
        debate_id: str,
        debate_result: Any,
        task: str,
    ) -> dict[str, Any] | None:
        """Step 1: Generate decision explanation."""
        try:
            from aragora.explainability.builder import ExplanationBuilder

            builder = ExplanationBuilder()

            # Extract key data from debate result
            consensus = getattr(debate_result, "consensus", None)
            messages = getattr(debate_result, "messages", [])

            explanation = builder.build(
                query=task,
                answer=str(consensus) if consensus else "",
                messages=messages if isinstance(messages, list) else [],
            )

            logger.info("Post-debate explanation generated for %s", debate_id)
            return {
                "debate_id": debate_id,
                "explanation": explanation,
                "task": task,
            }
        except ImportError:
            logger.debug("ExplanationBuilder not available")
            return None
        except Exception as e:
            logger.warning("Explanation generation failed: %s", e)
            return None

    def _step_create_plan(
        self,
        debate_id: str,
        debate_result: Any,
        task: str,
        explanation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Step 2: Create decision plan, enriched with explanation context."""
        try:
            from aragora.pipeline.decision_plan.core import DecisionPlanFactory

            factory = DecisionPlanFactory()

            # Include explanation in plan context if available
            context = {"debate_id": debate_id, "task": task}
            if explanation:
                context["explanation"] = explanation.get("explanation", "")

            plan = factory.create_from_debate(
                debate_id=debate_id,
                task=task,
                result=debate_result,
                context=context,
            )

            logger.info("Decision plan created for %s", debate_id)
            return {
                "debate_id": debate_id,
                "plan": plan,
                "has_explanation_context": explanation is not None,
            }
        except ImportError:
            logger.debug("DecisionPlanFactory not available")
            return None
        except Exception as e:
            logger.warning("Plan creation failed: %s", e)
            return None

    def _step_notify(
        self,
        debate_id: str,
        debate_result: Any,
        explanation: dict[str, Any] | None,
        plan: dict[str, Any] | None,
    ) -> bool:
        """Step 3: Send notifications with full context from prior steps."""
        try:
            from aragora.notifications.service import notify_debate_completed

            # Build notification with context from prior steps
            extra_context = {}
            if explanation:
                extra_context["has_explanation"] = True
            if plan:
                extra_context["has_plan"] = True
                plan_obj = plan.get("plan")
                if plan_obj and hasattr(plan_obj, "status"):
                    extra_context["plan_status"] = str(plan_obj.status)

            notify_debate_completed(
                debate_id=debate_id,
                task=getattr(debate_result, "task", ""),
                consensus_reached=bool(getattr(debate_result, "consensus", None)),
                confidence=getattr(debate_result, "confidence", 0.0),
            )

            logger.info("Post-debate notification sent for %s", debate_id)
            return True
        except ImportError:
            logger.debug("Notification service not available")
            return False
        except Exception as e:
            logger.warning("Notification failed: %s", e)
            return False

    def _step_execute_plan(
        self,
        plan_data: dict[str, Any],
        explanation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Step 4: Execute approved plan, using explanation as issue body context."""
        try:
            from aragora.pipeline.executor import PlanExecutor

            executor = PlanExecutor()
            plan_obj = plan_data.get("plan")
            if not plan_obj:
                return None

            # Check if plan is approved
            status = getattr(plan_obj, "status", None)
            if status and str(status).lower() != "approved":
                logger.info("Plan not approved, skipping execution")
                return {"skipped": True, "reason": f"status={status}"}

            result = executor.execute_to_github_issue(plan_obj)
            logger.info("Plan executed: %s", result)
            return result
        except ImportError:
            logger.debug("PlanExecutor not available")
            return None
        except Exception as e:
            logger.warning("Plan execution failed: %s", e)
            return None


__all__ = [
    "PostDebateCoordinator",
    "PostDebateConfig",
    "PostDebateResult",
]

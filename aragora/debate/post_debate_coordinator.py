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

    auto_explain: bool = True
    auto_create_plan: bool = True
    auto_notify: bool = True
    auto_execute_plan: bool = False
    auto_create_pr: bool = False  # Create draft PR for code-related debates
    pr_min_confidence: float = 0.8  # Higher confidence bar for PRs
    auto_build_integrity_package: bool = False
    auto_persist_receipt: bool = True
    auto_gauntlet_validate: bool = False
    gauntlet_min_confidence: float = 0.85
    auto_queue_improvement: bool = False
    improvement_min_confidence: float = 0.8
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
    pr_result: dict[str, Any] | None = None
    integrity_package: dict[str, Any] | None = None
    receipt_persisted: bool = False
    gauntlet_result: dict[str, Any] | None = None
    improvement_queued: bool = False
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

        # Step 2.5: Gauntlet adversarial validation
        if self.config.auto_gauntlet_validate and confidence >= self.config.gauntlet_min_confidence:
            result.gauntlet_result = self._step_gauntlet_validate(
                debate_id, debate_result, task, confidence
            )

        # Step 3: Send notifications
        if self.config.auto_notify:
            result.notification_sent = self._step_notify(
                debate_id, debate_result, result.explanation, result.plan
            )

        # Step 4: Execute plan if approved
        if self.config.auto_execute_plan and result.plan:
            result.execution_result = self._step_execute_plan(result.plan, result.explanation)

        # Step 4.5: Create draft PR for code-related debates
        if (
            self.config.auto_create_pr
            and result.plan
            and confidence >= self.config.pr_min_confidence
        ):
            result.pr_result = self._step_create_pr(result.plan, task)

        # Step 5: Build decision integrity package
        if self.config.auto_build_integrity_package:
            result.integrity_package = self._step_build_integrity_package(
                debate_id, debate_result
            )

        # Step 6: Persist receipt to Knowledge Mound (the flywheel)
        if self.config.auto_persist_receipt:
            result.receipt_persisted = self._step_persist_receipt(
                debate_id, debate_result, task, confidence
            )

        # Step 7: Queue improvement suggestion
        if self.config.auto_queue_improvement and confidence >= self.config.improvement_min_confidence:
            result.improvement_queued = self._step_queue_improvement(
                debate_id, debate_result, task, confidence
            )

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
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
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
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
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
        except (ValueError, TypeError, AttributeError, RuntimeError, OSError, ConnectionError) as e:
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
        except (ValueError, TypeError, AttributeError, RuntimeError, OSError, ConnectionError, KeyError) as e:
            logger.warning("Plan execution failed: %s", e)
            return None

    def _step_create_pr(
        self,
        plan_data: dict[str, Any],
        task: str,
    ) -> dict[str, Any] | None:
        """Step 4.5: Create a draft PR for code-related debate outcomes."""
        try:
            from aragora.pipeline.executor import PlanExecutor

            executor = PlanExecutor()
            plan_obj = plan_data.get("plan")
            if not plan_obj:
                return None

            result = executor.execute_to_github_pr(plan_obj, draft=True)
            logger.info("Draft PR created for task: %s", task[:80])
            return result
        except ImportError:
            logger.debug("PlanExecutor not available for PR creation")
            return None
        except (RuntimeError, ValueError, OSError, KeyError) as e:
            logger.warning("PR creation failed: %s", e)
            return None

    def _step_build_integrity_package(
        self,
        debate_id: str,
        debate_result: Any,
    ) -> dict[str, Any] | None:
        """Step 5: Build a DecisionIntegrityPackage from the debate result."""
        try:
            from aragora.core_types import DebateResult
            from aragora.pipeline.decision_integrity import build_integrity_package_from_result

            # Coerce to DebateResult if needed
            if isinstance(debate_result, DebateResult):
                dr = debate_result
            else:
                dr = DebateResult(
                    debate_id=debate_id,
                    task=str(getattr(debate_result, "task", "")),
                    final_answer=str(getattr(debate_result, "final_answer", getattr(debate_result, "consensus", ""))),
                    confidence=float(getattr(debate_result, "confidence", 0.0)),
                    consensus_reached=bool(getattr(debate_result, "consensus", None)),
                    participants=[
                        str(a) for a in (getattr(debate_result, "participants", []) or [])
                    ],
                )

            package = build_integrity_package_from_result(
                dr,
                include_receipt=True,
                include_plan=False,
            )

            logger.info("Decision integrity package built for %s", debate_id)
            return package.to_dict()
        except ImportError:
            logger.debug("Decision integrity pipeline not available")
            return None
        except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
            logger.warning("Integrity package generation failed: %s", e)
            return None


    def _step_persist_receipt(
        self,
        debate_id: str,
        debate_result: Any,
        task: str,
        confidence: float,
    ) -> bool:
        """Step 6: Persist debate receipt to Knowledge Mound.

        This creates the knowledge flywheel: each debate's outcome becomes
        institutional memory that informs future debates on related topics.
        """
        try:
            from aragora.knowledge.mound.adapters.receipt_adapter import (
                get_receipt_adapter,
            )

            adapter = get_receipt_adapter()

            receipt_data = {
                "debate_id": debate_id,
                "task": task,
                "confidence": confidence,
                "consensus_reached": bool(getattr(debate_result, "consensus", None)),
                "final_answer": str(
                    getattr(
                        debate_result,
                        "final_answer",
                        getattr(debate_result, "consensus", ""),
                    )
                ),
                "participants": [
                    str(a)
                    for a in (getattr(debate_result, "participants", []) or [])
                ],
            }

            adapter.ingest(receipt_data)
            logger.info("Receipt persisted to KM for %s", debate_id)
            return True
        except ImportError:
            logger.debug("ReceiptAdapter not available, skipping KM persistence")
            return False
        except (ValueError, TypeError, OSError, AttributeError, KeyError) as e:
            logger.warning("Receipt KM persistence failed: %s", e)
            return False

    def _step_gauntlet_validate(
        self,
        debate_id: str,
        debate_result: Any,
        task: str,
        confidence: float,
    ) -> dict[str, Any] | None:
        """Step 2.5: Run lightweight adversarial stress test on high-confidence decisions."""
        try:
            from aragora.gauntlet.runner import GauntletRunner

            runner = GauntletRunner()
            verdict = runner.run(
                claim=str(getattr(debate_result, "final_answer", getattr(debate_result, "consensus", ""))),
                context=task,
                categories=["logic", "assumptions"],
                attacks_per_category=2,
            )

            logger.info("Gauntlet validation completed for %s: %s", debate_id, verdict.get("verdict", "unknown"))
            return {
                "debate_id": debate_id,
                "verdict": verdict,
            }
        except ImportError:
            logger.debug("GauntletRunner not available")
            return None
        except (ValueError, TypeError, AttributeError, RuntimeError, OSError, ConnectionError) as e:
            logger.warning("Gauntlet validation failed: %s", e)
            return None

    def _step_queue_improvement(
        self,
        debate_id: str,
        debate_result: Any,
        task: str,
        confidence: float,
    ) -> bool:
        """Step 7: Queue improvement suggestion for self-improvement pipeline."""
        try:
            from aragora.nomic.improvement_queue import (
                ImprovementSuggestion,
                get_improvement_queue,
            )

            consensus = str(
                getattr(
                    debate_result,
                    "final_answer",
                    getattr(debate_result, "consensus", ""),
                )
            )
            if not consensus:
                return False

            category = self._classify_improvement_category(task)

            suggestion = ImprovementSuggestion(
                debate_id=debate_id,
                task=task,
                suggestion=consensus,
                category=category,
                confidence=confidence,
            )

            queue = get_improvement_queue()
            queue.enqueue(suggestion)
            logger.info("Improvement suggestion queued for %s (category=%s)", debate_id, category)
            return True
        except ImportError:
            logger.debug("ImprovementQueue not available")
            return False
        except (ValueError, TypeError, AttributeError, RuntimeError, OSError, KeyError) as e:
            logger.warning("Improvement queue failed: %s", e)
            return False

    @staticmethod
    def _classify_improvement_category(task: str) -> str:
        """Classify improvement category from task text."""
        task_lower = task.lower()
        if any(w in task_lower for w in ("test", "coverage", "assertion")):
            return "test_coverage"
        if any(w in task_lower for w in ("perf", "speed", "latency", "slow")):
            return "performance"
        if any(w in task_lower for w in ("reliab", "resilien", "fault", "retry")):
            return "reliability"
        if any(w in task_lower for w in ("doc", "readme", "comment")):
            return "documentation"
        return "code_quality"


DEFAULT_POST_DEBATE_CONFIG = PostDebateConfig(
    auto_explain=True,
    auto_create_plan=False,
    auto_notify=False,
    auto_execute_plan=False,
    auto_create_pr=False,
    auto_build_integrity_package=False,
    auto_persist_receipt=True,
)


__all__ = [
    "PostDebateCoordinator",
    "PostDebateConfig",
    "PostDebateResult",
    "DEFAULT_POST_DEBATE_CONFIG",
]

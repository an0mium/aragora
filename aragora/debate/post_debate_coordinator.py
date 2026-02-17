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
    auto_verify_arguments: bool = False
    auto_queue_improvement: bool = False
    improvement_min_confidence: float = 0.8
    plan_min_confidence: float = 0.7
    plan_approval_mode: str = "risk_based"
    # Calibration → blockchain reputation: push Brier scores to ERC-8004
    auto_push_calibration: bool = False
    calibration_min_predictions: int = 5  # Min predictions before pushing
    # Outcome feedback: feed systematic errors back to Nomic Loop
    auto_outcome_feedback: bool = False
    # Execution bridge: auto-trigger downstream actions
    auto_execution_bridge: bool = True
    execution_bridge_min_confidence: float = 0.0  # Bridge has per-rule thresholds


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
    argument_verification: dict[str, Any] | None = None
    improvement_queued: bool = False
    outcome_feedback: dict[str, Any] | None = None
    bridge_results: list[dict[str, Any]] = field(default_factory=list)
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

        # Step 2.7: Argument structure verification
        if self.config.auto_verify_arguments:
            result.argument_verification = self._step_argument_verification(
                debate_id, debate_result, task
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

        # Step 7.5: Push calibration data to ERC-8004 blockchain reputation
        if self.config.auto_push_calibration:
            self._step_push_calibration(debate_id, agents)

        # Step 7.7: Outcome feedback — feed systematic errors to Nomic Loop
        if self.config.auto_outcome_feedback:
            result.outcome_feedback = self._step_outcome_feedback(debate_id)

        # Step 8: Execution bridge — auto-trigger downstream actions
        if self.config.auto_execution_bridge and confidence >= self.config.execution_bridge_min_confidence:
            bridge_results = self._step_execution_bridge(
                debate_id, debate_result, task, confidence, agents,
            )
            result.bridge_results = bridge_results

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

    def _step_argument_verification(
        self,
        debate_id: str,
        debate_result: Any,
        task: str,
    ) -> dict[str, Any] | None:
        """Step 2.7: Verify logical structure of debate argument chains.

        Builds an argument graph from debate messages and runs formal
        verification to detect invalid chains, contradictions, circular
        dependencies, and unsupported conclusions.
        """
        try:
            from aragora.verification.argument_verifier import (
                ArgumentStructureVerifier,
            )
            from aragora.visualization.mapper import ArgumentCartographer
        except ImportError:
            logger.debug(f"ArgumentStructureVerifier not available for debate {debate_id}")
            return None

        try:
            import asyncio

            # Build argument graph from debate messages
            graph = ArgumentCartographer()
            graph.set_debate_context(debate_id, task)

            messages = getattr(debate_result, "messages", [])
            if isinstance(messages, list):
                for msg in messages:
                    agent = getattr(msg, "agent", "unknown")
                    content = getattr(msg, "content", "")
                    role = getattr(msg, "role", "proposal")
                    round_num = getattr(msg, "round", 0) or 0
                    if content:
                        graph.update_from_message(
                            agent=str(agent),
                            content=str(content),
                            role=str(role),
                            round_num=int(round_num),
                        )

            if not graph.nodes:
                logger.debug(f"No argument nodes to verify for debate {debate_id}")
                return None

            verifier = ArgumentStructureVerifier()
            verification_result = asyncio.run(verifier.verify(graph))

            logger.info(
                f"Argument verification completed for {debate_id}: "
                f"soundness={verification_result.soundness_score}"
            )
            return {
                "debate_id": debate_id,
                "verification": verification_result.to_dict(),
                "is_sound": verification_result.is_sound,
                "soundness_score": verification_result.soundness_score,
            }
        except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.warning(f"Argument verification failed for {debate_id}: {e}")
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

    def _step_execution_bridge(
        self,
        debate_id: str,
        debate_result: Any,
        task: str,
        confidence: float,
        agents: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Step 8: Run execution bridge to auto-trigger downstream actions."""
        try:
            from aragora.debate.execution_bridge import create_default_bridge

            bridge = create_default_bridge()
            agent_names = [
                getattr(a, "name", str(a)) for a in (agents or [])
            ]
            domain = "general"
            if hasattr(debate_result, "domain"):
                domain = debate_result.domain

            results = bridge.evaluate_and_execute(
                debate_id=debate_id,
                debate_result=debate_result,
                confidence=confidence,
                domain=domain,
                task=task,
                agents=agent_names,
            )

            executed = [r.to_dict() for r in results]
            if executed:
                logger.info(
                    "Execution bridge triggered %d actions for %s",
                    len(executed),
                    debate_id,
                )
            return executed
        except ImportError:
            logger.debug("ExecutionBridge not available")
            return []
        except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.warning("Execution bridge failed: %s", e)
            return []

    def _step_push_calibration(
        self,
        debate_id: str,
        agents: list[Any] | None = None,
    ) -> bool:
        """Step 7.5: Push agent calibration scores to ERC-8004 blockchain reputation.

        For each agent with sufficient prediction history, converts Brier score
        to a reputation signal and pushes it to the on-chain registry.
        """
        try:
            from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter

            adapter = ERC8004Adapter()
            pushed = 0

            for agent in (agents or []):
                agent_name = getattr(agent, "name", str(agent))
                calibration_tracker = getattr(agent, "calibration_tracker", None)
                if calibration_tracker is None:
                    continue

                # Get calibration data
                cal_data = None
                if hasattr(calibration_tracker, "get_calibration"):
                    cal_data = calibration_tracker.get_calibration(agent_name)
                elif hasattr(calibration_tracker, "get_calibration_score"):
                    cal_data = {"brier_score": calibration_tracker.get_calibration_score(agent_name)}

                if not cal_data:
                    continue

                prediction_count = cal_data.get("prediction_count", cal_data.get("count", 0))
                if prediction_count < self.config.calibration_min_predictions:
                    continue

                brier = cal_data.get("brier_score", cal_data.get("brier", 1.0))
                # Convert Brier score (0=perfect, 1=worst) to reputation (0-100)
                reputation = max(0, min(100, int((1.0 - brier) * 100)))

                adapter.push_reputation(
                    agent_id=agent_name,
                    score=reputation,
                    domain="calibration",
                    metadata={
                        "debate_id": debate_id,
                        "brier_score": brier,
                        "prediction_count": prediction_count,
                    },
                )
                pushed += 1

            if pushed:
                logger.info(
                    "Pushed calibration reputation for %d agents (debate %s)",
                    pushed,
                    debate_id,
                )
            return pushed > 0
        except ImportError:
            logger.debug("ERC8004Adapter not available, skipping calibration push")
            return False
        except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.warning("Calibration push failed: %s", e)
            return False

    def _step_outcome_feedback(
        self,
        debate_id: str,
    ) -> dict[str, Any] | None:
        """Step 7.7: Run outcome feedback cycle to detect systematic errors.

        Analyzes outcome patterns across past debates and queues
        improvement goals for the Nomic Loop MetaPlanner.
        """
        try:
            from aragora.nomic.outcome_feedback import OutcomeFeedbackBridge

            bridge = OutcomeFeedbackBridge()
            cycle_result = bridge.run_feedback_cycle()

            if cycle_result.get("goals_generated", 0) > 0:
                logger.info(
                    "Outcome feedback: %d goals generated, %d queued (debate %s)",
                    cycle_result["goals_generated"],
                    cycle_result["suggestions_queued"],
                    debate_id,
                )
            return cycle_result
        except ImportError:
            logger.debug("OutcomeFeedbackBridge not available, skipping outcome feedback")
            return None
        except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.debug("Outcome feedback failed (non-critical): %s", e)
            return None

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
    auto_gauntlet_validate=True,
    auto_push_calibration=True,
)


__all__ = [
    "PostDebateCoordinator",
    "PostDebateConfig",
    "PostDebateResult",
    "DEFAULT_POST_DEBATE_CONFIG",
]

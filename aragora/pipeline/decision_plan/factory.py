"""
DecisionPlan factory - creates plans from debate results.

Stability: STABLE
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from aragora.core_types import DebateResult
from aragora.implement.types import ImplementPlan, ImplementTask
from aragora.pipeline.decision_plan.core import (
    ApprovalMode,
    BudgetAllocation,
    DecisionPlan,
    PlanStatus,
)
from aragora.pipeline.risk_register import Risk, RiskCategory, RiskLevel, RiskRegister
from aragora.pipeline.verification_plan import (
    CasePriority,
    VerificationCase,
    VerificationPlan,
    VerificationType,
)


class DecisionPlanFactory:
    """Factory for creating DecisionPlan from DebateResult.

    Generates all sub-artifacts (risk register, verification plan,
    implementation plan) from the debate result and its metadata.

    Usage:
        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=5.00,
            approval_mode=ApprovalMode.RISK_BASED,
        )
    """

    @staticmethod
    def from_debate_result(
        result: DebateResult,
        *,
        budget_limit_usd: float | None = None,
        approval_mode: ApprovalMode = ApprovalMode.RISK_BASED,
        max_auto_risk: RiskLevel = RiskLevel.LOW,
        repo_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
        implement_plan: ImplementPlan | None = None,
    ) -> DecisionPlan:
        """Create a DecisionPlan from a DebateResult.

        This is the primary entry point for the gold path. It:
        1. Analyzes the debate result for risks
        2. Generates a verification plan
        3. Decomposes the conclusion into implementation tasks
        4. Sets up budget tracking from debate costs
        5. Configures approval based on risk assessment

        Args:
            result: The completed DebateResult from Arena.run().
            budget_limit_usd: Optional budget cap for the full plan.
            approval_mode: How human approval is determined.
            max_auto_risk: Max risk level for auto-execution.
            repo_path: Repository root for implementation planning.
            metadata: Additional metadata to attach.
            implement_plan: Optional pre-built implementation plan to reuse.

        Returns:
            A fully populated DecisionPlan ready for approval/execution.
        """
        plan = DecisionPlan(
            debate_id=result.debate_id,
            task=result.task,
            debate_result=result,
            approval_mode=approval_mode,
            max_auto_risk=max_auto_risk,
            metadata=metadata or {},
        )

        # Budget setup
        plan.budget = BudgetAllocation(
            limit_usd=budget_limit_usd,
            debate_cost_usd=result.total_cost_usd,
            spent_usd=result.total_cost_usd,
        )

        # Risk analysis from debate result
        plan.risk_register = DecisionPlanFactory._build_risk_register(result)

        # Verification plan from debate result
        plan.verification_plan = DecisionPlanFactory._build_verification_plan(result)

        # Implementation plan from debate conclusion (or reuse provided plan)
        if implement_plan is not None:
            plan.implement_plan = implement_plan
        else:
            plan.implement_plan = DecisionPlanFactory._build_implement_plan(result, repo_path)

        # Set status based on approval needs
        if plan.requires_human_approval:
            plan.status = PlanStatus.AWAITING_APPROVAL
        else:
            plan.status = PlanStatus.APPROVED

        return plan

    @staticmethod
    def from_implement_plan(
        implement_plan: ImplementPlan,
        *,
        debate_id: str = "",
        task: str = "",
    ) -> DecisionPlan:
        """Wrap an existing ImplementPlan as a DecisionPlan for persistence.

        Used when an ImplementPlan was created directly (e.g. via
        create_single_task_plan) and needs to be stored in the plan store.
        """
        return DecisionPlan(
            debate_id=debate_id,
            task=task or "Implementation plan from decision integrity package",
            implement_plan=implement_plan,
            status=PlanStatus.CREATED,
        )

    @staticmethod
    def _build_risk_register(result: DebateResult) -> RiskRegister:
        """Build risk register directly from DebateResult."""
        register = RiskRegister(debate_id=result.debate_id)

        # Low confidence → risk
        if result.confidence < 0.7:
            register.add_risk(
                Risk(
                    id=f"risk-confidence-{result.debate_id[:8]}",
                    title="Low consensus confidence",
                    description=(
                        f"Debate reached {result.confidence:.0%} confidence. "
                        "Implementation may face challenges or require revision."
                    ),
                    level=RiskLevel.MEDIUM if result.confidence >= 0.5 else RiskLevel.HIGH,
                    category=RiskCategory.UNKNOWN,
                    source="consensus_analysis",
                    impact=0.6,
                    likelihood=1.0 - result.confidence,
                )
            )

        # No consensus → risk
        if not result.consensus_reached:
            register.add_risk(
                Risk(
                    id=f"risk-no-consensus-{result.debate_id[:8]}",
                    title="No consensus reached",
                    description="Agents did not reach consensus. Decision may be contested.",
                    level=RiskLevel.HIGH,
                    category=RiskCategory.UNKNOWN,
                    source="consensus_analysis",
                    impact=0.8,
                    likelihood=0.7,
                )
            )

        # High-severity critiques → risks
        for i, critique in enumerate(result.critiques):
            if critique.severity >= 7.0:
                for j, issue in enumerate(critique.issues[:2]):
                    register.add_risk(
                        Risk(
                            id=f"risk-critique-{i}-{j}",
                            title=issue[:60],
                            description=issue,
                            level=RiskLevel.HIGH if critique.severity >= 8.0 else RiskLevel.MEDIUM,
                            category=_categorize_issue(issue),
                            source=f"critique:{critique.agent}",
                            impact=critique.severity / 10.0,
                            likelihood=0.7,
                            mitigation=", ".join(critique.suggestions[:2]),
                        )
                    )

        # Dissenting views → risks
        for i, view in enumerate(result.dissenting_views[:3]):
            register.add_risk(
                Risk(
                    id=f"risk-dissent-{i}",
                    title=f"Dissenting view: {view[:50]}",
                    description=view,
                    level=RiskLevel.MEDIUM,
                    category=RiskCategory.UNKNOWN,
                    source="dissent_analysis",
                    impact=0.5,
                    likelihood=0.4,
                )
            )

        # Debate cruxes → risks
        for i, crux in enumerate(result.debate_cruxes[:3]):
            claim = crux.get("claim", crux.get("text", "Unknown crux"))
            register.add_risk(
                Risk(
                    id=f"risk-crux-{i}",
                    title=f"Unresolved crux: {str(claim)[:50]}",
                    description=f"Key disagreement driver: {claim}",
                    level=RiskLevel.MEDIUM,
                    category=RiskCategory.TECHNICAL,
                    source="belief_network",
                    impact=0.5,
                    likelihood=0.5,
                )
            )

        return register

    @staticmethod
    def _build_verification_plan(result: DebateResult) -> VerificationPlan:
        """Build verification plan directly from DebateResult."""
        plan = VerificationPlan(
            debate_id=result.debate_id,
            title=f"Verify: {result.task[:60]}",
            description=f"Verification plan for debate {result.debate_id}",
        )

        # Extract testable claims from final answer
        test_num = 1
        if result.final_answer:
            for line in result.final_answer.split("\n"):
                line = line.strip()
                if not line or len(line) < 15:
                    continue
                keywords = ["implement", "use", "add", "create", "ensure", "should", "must"]
                if any(kw in line.lower() for kw in keywords):
                    plan.add_test(
                        VerificationCase(
                            id=f"consensus-{test_num}",
                            title=f"Verify: {line[:50]}",
                            description=f"Confirm implementation satisfies: {line}",
                            test_type=VerificationType.INTEGRATION,
                            priority=CasePriority.P1,
                            steps=[
                                "Set up environment",
                                "Execute functionality",
                                "Verify expected behavior",
                            ],
                            expected_result="Functionality works as described",
                        )
                    )
                    test_num += 1
                    if test_num > 5:
                        break

        # Edge cases from high-severity critiques
        for i, critique in enumerate(result.critiques[:5]):
            if critique.severity >= 5.0:
                for j, issue in enumerate(critique.issues[:1]):
                    plan.add_test(
                        VerificationCase(
                            id=f"critique-edge-{i}-{j}",
                            title=f"Edge case: {issue[:50]}",
                            description=f"Verify handling of: {issue}",
                            test_type=VerificationType.UNIT,
                            priority=CasePriority.P2,
                            steps=["Set up edge case", "Execute", "Verify graceful handling"],
                            expected_result="Edge case handled",
                        )
                    )

        # Smoke test
        plan.add_test(
            VerificationCase(
                id="smoke-1",
                title="Smoke test: Basic functionality",
                description="Verify basic functionality after implementation",
                test_type=VerificationType.E2E,
                priority=CasePriority.P0,
                steps=["Deploy changes", "Execute happy path", "Verify success"],
                expected_result="Basic use case succeeds",
            )
        )

        # Regression
        plan.add_test(
            VerificationCase(
                id="regression-1",
                title="Regression: Existing functionality",
                description="Verify no regressions in existing functionality",
                test_type=VerificationType.REGRESSION,
                priority=CasePriority.P1,
                steps=["Run existing test suite", "Verify all pass"],
                expected_result="No regressions",
            )
        )

        return plan

    @staticmethod
    def _build_implement_plan(result: DebateResult, repo_path: Path | None = None) -> ImplementPlan:
        """Build implementation plan from debate conclusion.

        Uses heuristic extraction from the final answer. For richer
        decomposition, callers should use generate_implement_plan()
        from aragora.implement.planner with an LLM.
        """
        design = result.final_answer or result.task
        design_hash = hashlib.sha256(design.encode()).hexdigest()

        tasks: list[ImplementTask] = []
        task_num = 1

        # Extract numbered steps from the final answer
        if result.final_answer:
            for line in result.final_answer.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Match numbered items or bullet points with action verbs
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                    clean = line.lstrip("0123456789.-*) ").strip()
                    if len(clean) > 15:
                        # Infer file paths mentioned
                        import re

                        files = re.findall(r"`([a-zA-Z0-9_/\-\.]+\.[a-z]+)`", clean)

                        tasks.append(
                            ImplementTask(
                                id=f"task-{task_num}",
                                description=clean[:200],
                                files=files[:5],
                                complexity="moderate",
                                dependencies=[f"task-{task_num - 1}"] if task_num > 1 else [],
                            )
                        )
                        task_num += 1
                        if task_num > 10:
                            break

        # Fallback: single task if no structured steps found
        if not tasks:
            tasks.append(
                ImplementTask(
                    id="task-1",
                    description="Implement the debated solution",
                    files=[],
                    complexity="complex",
                    dependencies=[],
                )
            )

        return ImplementPlan(design_hash=design_hash, tasks=tasks)

    @staticmethod
    async def from_debate_result_async(
        result: DebateResult,
        *,
        knowledge_mound: Any | None = None,
        budget_limit_usd: float | None = None,
        approval_mode: ApprovalMode = ApprovalMode.RISK_BASED,
        max_auto_risk: RiskLevel = RiskLevel.LOW,
        repo_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
        implement_plan: ImplementPlan | None = None,
        enrich_from_history: bool = True,
    ) -> DecisionPlan:
        """Async version of from_debate_result with Knowledge Mound enrichment.

        This extended factory method queries the Knowledge Mound for historical
        decisions similar to the current task, enriching the risk register with
        data about past outcomes.

        Args:
            result: The completed DebateResult from Arena.run().
            knowledge_mound: Optional KM instance for historical queries.
            budget_limit_usd: Optional budget cap for the full plan.
            approval_mode: How human approval is determined.
            max_auto_risk: Max risk level for auto-execution.
            repo_path: Repository root for implementation planning.
            metadata: Additional metadata to attach.
            implement_plan: Optional pre-built implementation plan to reuse.
            enrich_from_history: Whether to query KM for historical context.

        Returns:
            A DecisionPlan enriched with historical risk data.
        """
        # Start with synchronous creation
        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=budget_limit_usd,
            approval_mode=approval_mode,
            max_auto_risk=max_auto_risk,
            repo_path=repo_path,
            metadata=metadata,
            implement_plan=implement_plan,
        )

        # Enrich risks with historical data from Knowledge Mound
        if enrich_from_history and plan.risk_register:
            await DecisionPlanFactory._enrich_risks_from_history(
                plan.risk_register, result.task, knowledge_mound
            )

        return plan

    @staticmethod
    async def _enrich_risks_from_history(
        register: RiskRegister,
        task: str,
        knowledge_mound: Any | None = None,
    ) -> None:
        """Enrich risk register with historical data from Knowledge Mound.

        Queries KM for similar past decisions and their outcomes, updating
        each risk with historical context (how often similar risks appeared,
        what the success rates were, etc.).

        Args:
            register: The risk register to enrich
            task: The task description for similarity search
            knowledge_mound: Optional KM instance (uses global if not provided)
        """
        try:
            from aragora.knowledge.mound.adapters.decision_plan_adapter import (
                get_decision_plan_adapter,
            )

            adapter = get_decision_plan_adapter(knowledge_mound)

            # Query for similar historical plans
            similar_plans = await adapter.query_similar_plans(task, limit=10)
            if not similar_plans:
                return

            # Aggregate historical data
            total_plans = len(similar_plans)
            successful_plans = sum(1 for p in similar_plans if p.get("success", False))
            failed_plans = total_plans - successful_plans
            overall_success_rate = successful_plans / total_plans if total_plans > 0 else None

            # For each risk, try to find similar patterns in historical plans
            for risk in register.risks:
                # Find plans where similar issues appeared (keyword matching)
                risk_keywords = _extract_keywords(risk.title + " " + risk.description)
                matching_plans: list[dict[str, Any]] = []

                for plan_data in similar_plans:
                    content = plan_data.get("content", "")
                    plan_task = plan_data.get("task", "")

                    # Check if risk keywords appear in historical plan
                    if any(kw.lower() in (content + plan_task).lower() for kw in risk_keywords):
                        matching_plans.append(plan_data)

                if matching_plans:
                    # Update risk with historical data
                    risk.historical_occurrences = len(matching_plans)
                    matching_successes = sum(1 for p in matching_plans if p.get("success", False))
                    risk.historical_success_rate = (
                        matching_successes / len(matching_plans) if matching_plans else None
                    )
                    risk.related_plan_ids = [
                        p.get("plan_id", "") for p in matching_plans if p.get("plan_id")
                    ][:5]

                    # Adjust likelihood based on historical failure rate
                    if risk.historical_success_rate is not None:
                        failure_rate = 1.0 - risk.historical_success_rate
                        # Blend historical failure rate with original estimate
                        risk.likelihood = (risk.likelihood + failure_rate) / 2

            # Add a meta-risk if overall success rate is low
            if overall_success_rate is not None and overall_success_rate < 0.7:
                from aragora.pipeline.risk_register import Risk, RiskCategory, RiskLevel

                register.add_risk(
                    Risk(
                        id=f"risk-history-{register.debate_id[:8]}",
                        title="Historical pattern: Similar tasks had low success",
                        description=(
                            f"Analysis of {total_plans} similar historical decisions shows "
                            f"{overall_success_rate:.0%} success rate. "
                            f"{failed_plans} similar tasks failed previously."
                        ),
                        level=RiskLevel.HIGH if overall_success_rate < 0.5 else RiskLevel.MEDIUM,
                        category=RiskCategory.UNKNOWN,
                        source="knowledge_mound",
                        impact=0.7,
                        likelihood=1.0 - overall_success_rate,
                        historical_occurrences=total_plans,
                        historical_success_rate=overall_success_rate,
                    )
                )

        except ImportError as e:
            # KM adapter not available, skip enrichment
            import logging

            logging.getLogger(__name__).debug("KM enrichment unavailable: %s", e)
        except (OSError, RuntimeError, ValueError) as e:
            import logging

            logging.getLogger(__name__).warning("Historical enrichment failed: %s", e)


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text for matching."""

    # Remove common words and extract meaningful tokens
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "and",
        "or",
        "but",
        "if",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "not",
        "no",
        "yes",
    }

    # Tokenize and filter
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    keywords = [t for t in tokens if t not in stop_words and len(t) > 3]

    # Return unique keywords (preserve order)
    seen: set[str] = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:10]  # Limit to 10 keywords


def _categorize_issue(issue: str) -> RiskCategory:
    """Categorize a risk issue by keywords."""
    lower = issue.lower()
    if any(k in lower for k in ["security", "auth", "permission", "vulnerable", "injection"]):
        return RiskCategory.SECURITY
    if any(k in lower for k in ["performance", "slow", "latency", "speed", "timeout"]):
        return RiskCategory.PERFORMANCE
    if any(k in lower for k in ["scale", "load", "capacity", "throughput"]):
        return RiskCategory.SCALABILITY
    if any(k in lower for k in ["maintain", "complex", "readab", "test", "debt"]):
        return RiskCategory.MAINTAINABILITY
    if any(k in lower for k in ["compat", "version", "depend", "integrat", "migrat"]):
        return RiskCategory.COMPATIBILITY
    return RiskCategory.TECHNICAL

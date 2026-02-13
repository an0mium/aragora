"""
Decision Pipeline CLI commands.

Contains the 'decide' command that runs the full gold path:
debate → plan → approve → execute → verify → learn

And the 'plans' command for managing decision plans.
"""

import argparse
import asyncio
import logging
import sys
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)


async def run_decide(
    task: str,
    agents_str: str,
    rounds: int = 9,
    context: str = "",
    documents: list[str] | None = None,
    auto_approve: bool = False,
    dry_run: bool = False,
    budget_limit: float | None = None,
    execution_mode: str | None = None,
    implementation_profile: dict[str, Any] | None = None,
    auto_select: bool = False,
    auto_select_config: dict[str, Any] | None = None,
    enable_knowledge_retrieval: bool | None = None,
    enable_knowledge_ingestion: bool | None = None,
    enable_cross_debate_memory: bool | None = None,
    enable_supermemory: bool | None = None,
    supermemory_context_container_tag: str | None = None,
    supermemory_max_context_items: int | None = None,
    enable_belief_guidance: bool | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full decision pipeline: debate → plan → execute.

    Args:
        task: The task/question to decide on
        agents_str: Comma-separated agent specs
        rounds: Number of debate rounds
        auto_approve: Automatically approve plans (skip approval)
        dry_run: Create plan but don't execute
        budget_limit: Maximum budget for plan execution in USD
        execution_mode: Execution engine override ("workflow", "hybrid", "fabric", "computer_use")
        verbose: Print detailed progress

    Returns:
        Dict with debate_result, plan, outcome (if executed)
    """
    from aragora.cli.commands.debate import run_debate
    from aragora.pipeline.decision_plan import (
        ApprovalMode,
        DecisionPlanFactory,
    )
    from aragora.pipeline.executor import PlanExecutor, store_plan

    result: dict[str, Any] = {}

    # Step 1: Run the debate
    if verbose:
        print(f"[decide] Running debate with {agents_str}...")

    debate_result = await run_debate(
        task=task,
        agents_str=agents_str,
        rounds=rounds,
        context=context,
        documents=documents,
        auto_select=auto_select,
        auto_select_config=auto_select_config,
        enable_knowledge_retrieval=enable_knowledge_retrieval,
        enable_knowledge_ingestion=enable_knowledge_ingestion,
        enable_cross_debate_memory=enable_cross_debate_memory,
        enable_supermemory=enable_supermemory,
        supermemory_context_container_tag=supermemory_context_container_tag,
        supermemory_max_context_items=supermemory_max_context_items,
        enable_belief_guidance=enable_belief_guidance,
    )
    result["debate_result"] = debate_result

    if verbose:
        print(f"[decide] Debate complete. Consensus: {debate_result.consensus_reached}")
        print(f"[decide] Confidence: {debate_result.confidence:.1%}")

    # Step 2: Create decision plan
    if verbose:
        print("[decide] Creating decision plan...")

    approval_mode = ApprovalMode.NEVER if auto_approve else ApprovalMode.RISK_BASED

    plan = DecisionPlanFactory.from_debate_result(
        debate_result,
        budget_limit_usd=budget_limit,
        approval_mode=approval_mode,
        implementation_profile=implementation_profile,
    )
    store_plan(plan)
    result["plan"] = plan

    if verbose:
        print(f"[decide] Plan created: {plan.id[:12]}...")
        print(f"[decide] Plan status: {plan.status.value}")
        if plan.risk_register:
            print(f"[decide] Risk summary: {plan.risk_register.summary}")

    # Step 3: Handle approval
    if plan.requires_human_approval and not auto_approve:
        if verbose:
            print("[decide] Plan requires approval. Use --auto-approve to skip.")
            print(f"[decide] Run: aragora plans approve {plan.id}")
        result["requires_approval"] = True
        return result

    # Step 4: Execute if not dry run
    if dry_run:
        if verbose:
            print("[decide] Dry run mode - skipping execution.")
        result["dry_run"] = True
        return result

    if verbose:
        print("[decide] Executing plan...")

    ExecutionMode = Literal["workflow", "hybrid", "fabric", "computer_use"]
    _mode = cast(ExecutionMode | None, execution_mode)
    executor = PlanExecutor(execution_mode=_mode)

    try:
        outcome = await executor.execute(plan, execution_mode=_mode)
        result["outcome"] = outcome

        if verbose:
            print(f"[decide] Execution complete. Success: {outcome.success}")
            print(f"[decide] Tasks: {outcome.tasks_completed}/{outcome.tasks_total}")
            if outcome.receipt_id:
                print(f"[decide] Receipt: {outcome.receipt_id[:12]}...")
            if outcome.lessons:
                print("[decide] Lessons learned:")
                for lesson in outcome.lessons[:3]:
                    print(f"  - {lesson}")

    except ValueError as e:
        result["error"] = str(e)
        if verbose:
            print(f"[decide] Execution failed: {e}")

    return result


def cmd_decide(args: argparse.Namespace) -> None:
    """Handle 'decide' command - full gold path."""
    from aragora.cli.commands.debate import (
        _append_context_file,
        _parse_auto_select_config,
        _parse_document_ids,
    )
    from aragora.pipeline.decision_plan.factory import normalize_execution_mode

    execution_mode = getattr(args, "execution_mode", None)
    if getattr(args, "computer_use", False):
        execution_mode = "computer_use"
    elif getattr(args, "hybrid", False):
        execution_mode = "hybrid"
    elif getattr(args, "fabric", False):
        execution_mode = "fabric"
    execution_mode = normalize_execution_mode(execution_mode)

    implementation_profile = None
    raw_profile = getattr(args, "implementation_profile", None)
    if raw_profile:
        import json

        try:
            implementation_profile = json.loads(raw_profile)
        except json.JSONDecodeError as e:
            print(f"Invalid --implementation-profile JSON: {e}", file=sys.stderr)
            raise SystemExit(2)
        if not isinstance(implementation_profile, dict):
            print("--implementation-profile must be a JSON object", file=sys.stderr)
            raise SystemExit(2)
        implementation_profile["execution_mode"] = normalize_execution_mode(
            implementation_profile.get("execution_mode")
        )

    def _split_csv(raw: str | None) -> list[str] | None:
        if not raw:
            return None
        return [item.strip() for item in raw.split(",") if item.strip()]

    fabric_models = _split_csv(getattr(args, "fabric_models", None))
    channel_targets = _split_csv(getattr(args, "channel_targets", None))
    thread_id = getattr(args, "thread_id", None)
    raw_threads = getattr(args, "thread_id_by_platform", None)
    thread_id_by_platform = None
    if raw_threads:
        import json

        try:
            thread_id_by_platform = json.loads(raw_threads)
        except json.JSONDecodeError as e:
            print(f"Invalid --thread-id-by-platform JSON: {e}", file=sys.stderr)
            raise SystemExit(2)
        if not isinstance(thread_id_by_platform, dict):
            print("--thread-id-by-platform must be a JSON object", file=sys.stderr)
            raise SystemExit(2)

    if any([fabric_models, channel_targets, thread_id, thread_id_by_platform]):
        if implementation_profile is None:
            implementation_profile = {}
        if fabric_models and "fabric_models" not in implementation_profile:
            implementation_profile["fabric_models"] = fabric_models
        if channel_targets and "channel_targets" not in implementation_profile:
            implementation_profile["channel_targets"] = channel_targets
        if thread_id and "thread_id" not in implementation_profile:
            implementation_profile["thread_id"] = thread_id
        if thread_id_by_platform and "thread_id_by_platform" not in implementation_profile:
            implementation_profile["thread_id_by_platform"] = thread_id_by_platform

    if execution_mode:
        if implementation_profile is None:
            implementation_profile = {}
        implementation_profile.setdefault("execution_mode", execution_mode)

    auto_select = bool(getattr(args, "auto_select", False))
    try:
        auto_select_config = _parse_auto_select_config(getattr(args, "auto_select_config", None))
    except ValueError as e:
        print(f"Invalid --auto-select-config: {e}", file=sys.stderr)
        raise SystemExit(2)
    if auto_select_config and not auto_select:
        auto_select = True

    context = getattr(args, "context", None) or ""
    context_file = getattr(args, "context_file", None)
    if context_file:
        try:
            context = _append_context_file(context, context_file)
        except (OSError, UnicodeDecodeError, ValueError) as e:
            print(f"Failed to read --context-file: {e}", file=sys.stderr)
            raise SystemExit(2)

    documents = _parse_document_ids(
        getattr(args, "document", None),
        getattr(args, "documents", None),
    )

    no_knowledge = bool(getattr(args, "no_knowledge", False))
    no_cross_memory = bool(getattr(args, "no_cross_memory", False))
    enable_supermemory = bool(getattr(args, "enable_supermemory", False))
    supermemory_container = getattr(args, "supermemory_container", None)
    supermemory_max_items = getattr(args, "supermemory_max_items", None)
    enable_belief_guidance = bool(getattr(args, "enable_belief_guidance", False))

    if supermemory_container or supermemory_max_items is not None:
        enable_supermemory = True

    result = asyncio.run(
        run_decide(
            task=args.task,
            agents_str=args.agents,
            rounds=args.rounds,
            context=context,
            documents=documents or None,
            auto_approve=getattr(args, "auto_approve", False),
            dry_run=getattr(args, "dry_run", False),
            budget_limit=getattr(args, "budget_limit", None),
            execution_mode=execution_mode,
            implementation_profile=implementation_profile,
            auto_select=auto_select,
            auto_select_config=auto_select_config,
            enable_knowledge_retrieval=None if not no_knowledge else False,
            enable_knowledge_ingestion=None if not no_knowledge else False,
            enable_cross_debate_memory=None if not no_cross_memory else False,
            enable_supermemory=True if enable_supermemory else None,
            supermemory_context_container_tag=supermemory_container,
            supermemory_max_context_items=supermemory_max_items,
            enable_belief_guidance=True if enable_belief_guidance else None,
            verbose=getattr(args, "verbose", False),
        )
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)

    debate_result = result.get("debate_result")
    if debate_result:
        print(f"Task: {debate_result.task[:100]}...")
        print(f"Consensus: {'Reached' if debate_result.consensus_reached else 'Not reached'}")
        print(f"Confidence: {debate_result.confidence:.1%}")
        print()

    plan = result.get("plan")
    if plan:
        print(f"Plan ID: {plan.id}")
        print(f"Plan Status: {plan.status.value}")

    if result.get("requires_approval"):
        print("\nAction required: Plan needs approval before execution.")
        print(f"Run: aragora plans approve {plan.id if plan else '<plan_id>'}")
        sys.exit(0)

    if result.get("dry_run"):
        print("\nDry run complete. Plan created but not executed.")
        sys.exit(0)

    outcome = result.get("outcome")
    if outcome:
        print()
        print("EXECUTION RESULT:")
        print("-" * 40)
        print(f"Success: {outcome.success}")
        print(f"Tasks: {outcome.tasks_completed}/{outcome.tasks_total}")
        print(f"Verification: {outcome.verification_passed}/{outcome.verification_total}")
        print(f"Duration: {outcome.duration_seconds:.1f}s")
        if outcome.total_cost_usd > 0:
            print(f"Cost: ${outcome.total_cost_usd:.4f}")
        if outcome.error:
            print(f"Error: {outcome.error}")

    error = result.get("error")
    if error:
        print(f"\nExecution Error: {error}")
        sys.exit(1)


def cmd_plans(args: argparse.Namespace) -> None:
    """Handle 'plans' command - list plans."""
    from aragora.pipeline.decision_plan import PlanStatus
    from aragora.pipeline.executor import list_plans

    status_filter = None
    if hasattr(args, "status") and args.status:
        try:
            status_filter = PlanStatus(args.status)
        except ValueError:
            print(f"Invalid status: {args.status}")
            sys.exit(1)

    limit = getattr(args, "limit", 20)
    plans = list_plans(status=status_filter, limit=limit)

    if not plans:
        print("No plans found.")
        return

    print(f"{'ID':<12} {'Status':<18} {'Task':<40} {'Created':<20}")
    print("-" * 90)

    for plan in plans:
        plan_id = plan.id[:12]
        status = plan.status.value
        task = plan.task[:40] + "..." if len(plan.task) > 40 else plan.task
        created = plan.created_at.strftime("%Y-%m-%d %H:%M") if plan.created_at else "N/A"
        print(f"{plan_id:<12} {status:<18} {task:<40} {created:<20}")


def cmd_plans_show(args: argparse.Namespace) -> None:
    """Handle 'plans show <id>' command."""
    from aragora.pipeline.executor import get_outcome, get_plan

    plan = get_plan(args.plan_id)
    if not plan:
        print(f"Plan not found: {args.plan_id}")
        sys.exit(1)

    print(f"Plan ID: {plan.id}")
    print(f"Debate ID: {plan.debate_id}")
    print(f"Status: {plan.status.value}")
    print(f"Task: {plan.task}")
    print()

    if plan.risk_register:
        print("Risk Summary:")
        print(f"  {plan.risk_register.summary}")
        print()

    if plan.implement_plan and plan.implement_plan.tasks:
        print(f"Tasks ({len(plan.implement_plan.tasks)}):")
        for i, task in enumerate(plan.implement_plan.tasks[:5], 1):
            print(f"  {i}. {task.description[:60]}...")
        if len(plan.implement_plan.tasks) > 5:
            print(f"  ... and {len(plan.implement_plan.tasks) - 5} more")
        print()

    if plan.approval_record:
        print("Approval:")
        print(f"  Approved: {plan.approval_record.approved}")
        print(f"  Approver: {plan.approval_record.approver_id}")
        if plan.approval_record.reason:
            print(f"  Reason: {plan.approval_record.reason}")
        print()

    outcome = get_outcome(plan.id)
    if outcome:
        print("Outcome:")
        print(f"  Success: {outcome.success}")
        print(f"  Tasks: {outcome.tasks_completed}/{outcome.tasks_total}")
        print(f"  Duration: {outcome.duration_seconds:.1f}s")
        if outcome.receipt_id:
            print(f"  Receipt: {outcome.receipt_id}")


def cmd_plans_approve(args: argparse.Namespace) -> None:
    """Handle 'plans approve <id>' command."""
    from aragora.pipeline.decision_plan import PlanStatus
    from aragora.pipeline.executor import get_plan, store_plan

    plan = get_plan(args.plan_id)
    if not plan:
        print(f"Plan not found: {args.plan_id}")
        sys.exit(1)

    if plan.status not in (PlanStatus.CREATED, PlanStatus.AWAITING_APPROVAL):
        print(f"Plan cannot be approved in status: {plan.status.value}")
        sys.exit(1)

    reason = getattr(args, "reason", "") or ""
    plan.approve(approver_id="cli-user", reason=reason)
    store_plan(plan)

    print(f"Plan {plan.id[:12]}... approved.")


def cmd_plans_reject(args: argparse.Namespace) -> None:
    """Handle 'plans reject <id>' command."""
    from aragora.pipeline.decision_plan import PlanStatus
    from aragora.pipeline.executor import get_plan, store_plan

    plan = get_plan(args.plan_id)
    if not plan:
        print(f"Plan not found: {args.plan_id}")
        sys.exit(1)

    if plan.status not in (PlanStatus.CREATED, PlanStatus.AWAITING_APPROVAL):
        print(f"Plan cannot be rejected in status: {plan.status.value}")
        sys.exit(1)

    reason = getattr(args, "reason", "Rejected via CLI") or "Rejected via CLI"
    plan.reject(approver_id="cli-user", reason=reason)
    store_plan(plan)

    print(f"Plan {plan.id[:12]}... rejected.")


def cmd_plans_execute(args: argparse.Namespace) -> None:
    """Handle 'plans execute <id>' command."""
    from aragora.pipeline.executor import PlanExecutor, get_plan

    plan = get_plan(args.plan_id)
    if not plan:
        print(f"Plan not found: {args.plan_id}")
        sys.exit(1)

    execution_mode = getattr(args, "execution_mode", None)
    if getattr(args, "computer_use", False):
        execution_mode = "computer_use"
    elif getattr(args, "hybrid", False):
        execution_mode = "hybrid"

    executor = PlanExecutor(execution_mode=execution_mode)

    print(f"Executing plan {plan.id[:12]}...")

    try:
        outcome = asyncio.run(executor.execute(plan, execution_mode=execution_mode))
        print()
        print("Execution complete:")
        print(f"  Success: {outcome.success}")
        print(f"  Tasks: {outcome.tasks_completed}/{outcome.tasks_total}")
        print(f"  Duration: {outcome.duration_seconds:.1f}s")
        if outcome.receipt_id:
            print(f"  Receipt: {outcome.receipt_id}")
        if outcome.error:
            print(f"  Error: {outcome.error}")
    except ValueError as e:
        print(f"Execution failed: {e}")
        sys.exit(1)


__all__ = [
    "cmd_decide",
    "cmd_plans",
    "cmd_plans_show",
    "cmd_plans_approve",
    "cmd_plans_reject",
    "cmd_plans_execute",
    "run_decide",
]

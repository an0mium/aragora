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
    template: str | None = None,
    mode: str = "standard",
    verbose: bool = False,
    **kwargs: Any,
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

    # Apply template overrides if specified
    if template:
        try:
            from aragora.workflow.templates import (
                get_template as get_wf_template,
                WORKFLOW_TEMPLATES,
            )

            tmpl = get_wf_template(template)
            if tmpl is None:
                # Try prefix matching (e.g., "sme_decision" → factory templates)
                for key in WORKFLOW_TEMPLATES:
                    if template in key or key.endswith(template):
                        tmpl = WORKFLOW_TEMPLATES[key]
                        break
            if tmpl:
                # Override debate params from template
                if tmpl.get("agents"):
                    agents_str = (
                        ",".join(tmpl["agents"])
                        if isinstance(tmpl["agents"], list)
                        else tmpl["agents"]
                    )
                if tmpl.get("rounds"):
                    rounds = tmpl["rounds"]
                if verbose:
                    print(f"[decide] Using template: {template}")
            else:
                if verbose:
                    print(f"[decide] Template '{template}' not found, using defaults")
        except ImportError:
            if verbose:
                print("[decide] Template system not available")

    # Apply mode overrides
    mode_config: dict[str, Any] = {}
    if mode and mode != "standard":
        try:
            from aragora.modes import load_builtins
            from aragora.modes.base import ModeRegistry

            load_builtins()
            mode_def = ModeRegistry.get(mode)
            if mode_def:
                mode_config = {  # noqa: F841 — stored for future mode injection
                    "mode": mode,
                    "mode_definition": mode_def,
                    "mode_system_prompt": mode_def.get_system_prompt(),
                }
                if verbose:
                    print(f"[decide] Using mode: {mode}")
            else:
                available = ", ".join(ModeRegistry.list_all())
                raise KeyError(f"Mode '{mode}' not found. Available: {available}")
        except KeyError:
            raise
        except ImportError:
            if verbose:
                print(f"[decide] Mode system not available, ignoring --mode {mode}")

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
        **kwargs,
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


def _run_decide_demo(args: argparse.Namespace) -> None:
    """Run the decide command in demo mode using mock agents.

    Produces a decision summary and receipt without requiring API keys.
    """
    import json
    import time

    task = args.task
    verbose = getattr(args, "verbose", False)

    print("\n" + "=" * 60)
    print("  ARAGORA DECIDE (Demo Mode)")
    print("  Full decision pipeline with mock agents")
    print("=" * 60)
    print(f"\n  Task: {task}")
    print("  Mode: Offline (no API keys required)")
    print()

    # Step 1: Run debate via aragora-debate mock agents
    if verbose:
        print("[decide-demo] Running debate with mock agents...")

    from aragora_debate.arena import Arena
    from aragora_debate.styled_mock import StyledMockAgent
    from aragora_debate.types import DebateConfig

    agents = [
        StyledMockAgent("Analyst", style="supportive"),
        StyledMockAgent("Critic", style="critical"),
        StyledMockAgent("Synthesizer", style="balanced"),
        StyledMockAgent("Devil's Advocate", style="contrarian"),
    ]

    rounds = min(getattr(args, "rounds", 2), 3)
    config = DebateConfig(rounds=rounds, early_stopping=False)
    arena = Arena(question=task, agents=agents, config=config)

    start_time = time.monotonic()
    debate_result = asyncio.run(arena.run())
    elapsed = time.monotonic() - start_time

    # Step 2: Build receipt
    from aragora.cli.demo import _build_receipt_data
    from aragora.cli.receipt_formatter import receipt_to_html

    receipt_data = _build_receipt_data(debate_result, elapsed)

    # Print decision summary
    print("=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)

    verdict = receipt_data.get("verdict", "N/A")
    confidence = receipt_data.get("confidence", 0)
    receipt_id = receipt_data.get("receipt_id", "")

    print(f"Task: {task}")
    print(f"Verdict: {verdict}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Agents: {', '.join(receipt_data.get('agents', []))}")
    print(f"Rounds: {receipt_data.get('rounds', 0)}")
    print(f"Duration: {elapsed:.2f}s")
    if receipt_id:
        print(f"Receipt ID: {receipt_id}")

    if debate_result.final_answer:
        print()
        print("WINNING POSITION:")
        print("-" * 40)
        print(debate_result.final_answer[:500])

    if receipt_data.get("consensus_proof"):
        cp = receipt_data["consensus_proof"]
        print()
        print("CONSENSUS:")
        print("-" * 40)
        print(f"  Reached: {'Yes' if cp.get('reached') else 'No'}")
        print(f"  Method: {cp.get('method', 'N/A')}")
        print(f"  Confidence: {cp.get('confidence', 0):.0%}")

    # Step 3: Save receipt
    dry_run = getattr(args, "dry_run", False)
    if not dry_run:
        from pathlib import Path

        receipts_dir = Path.cwd() / ".aragora" / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)
        receipt_file = receipts_dir / "decide-demo-receipt.json"
        receipt_file.write_text(json.dumps(receipt_data, indent=2, default=str))

        html_file = receipts_dir / "decide-demo-receipt.html"
        html_file.write_text(receipt_to_html(receipt_data))

        print()
        print(f"Receipt (JSON): {receipt_file}")
        print(f"Receipt (HTML): {html_file}")
        print()
        print("View receipt: aragora receipt view " + str(html_file))

    print()
    print("DEMO NOTE:")
    print("-" * 40)
    print("  This used mock agents. For real AI-powered decisions:")
    if dry_run:
        print("  (Dry run mode - no receipt saved)")
    print()
    print(f'  aragora decide "{task}" --agents anthropic-api,openai-api')
    print()


def cmd_decide(args: argparse.Namespace) -> None:
    """Handle 'decide' command - full gold path."""
    # Handle --list-templates
    if getattr(args, "list_templates", False):
        _print_available_templates()
        return

    # Handle --demo: run offline debate with mock agents, produce receipt
    if getattr(args, "demo", False):
        _run_decide_demo(args)
        return

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

    auto_select = bool(getattr(args, "auto_select", True))
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

    # Build MemoryConfig from CLI args (replaces individual deprecated params)
    from aragora.debate.arena_primary_configs import MemoryConfig

    no_knowledge = bool(getattr(args, "no_knowledge", False))
    no_cross_memory = bool(getattr(args, "no_cross_memory", False))
    enable_supermemory = bool(getattr(args, "enable_supermemory", False))
    supermemory_container = getattr(args, "supermemory_container", None)
    supermemory_max_items = getattr(args, "supermemory_max_items", None)
    enable_belief_guidance = bool(getattr(args, "enable_belief_guidance", False))

    if supermemory_container or supermemory_max_items is not None:
        enable_supermemory = True

    memory_config = MemoryConfig(
        enable_knowledge_retrieval=not no_knowledge,
        enable_knowledge_ingestion=not no_knowledge,
        enable_cross_debate_memory=not no_cross_memory,
        enable_supermemory=enable_supermemory,
        enable_belief_guidance=enable_belief_guidance,
    )
    if supermemory_container:
        memory_config.supermemory_context_container_tag = supermemory_container
    if supermemory_max_items is not None:
        memory_config.supermemory_max_context_items = supermemory_max_items

    # Apply preset configuration if specified
    extra_kwargs: dict[str, Any] = {}
    preset_name = getattr(args, "preset", None)
    if preset_name:
        from aragora.debate.presets import get_preset

        extra_kwargs.update(get_preset(preset_name))
        print(f"[preset] Applied '{preset_name}' configuration preset")

    # Create spectator stream if --spectate is specified
    if getattr(args, "spectate", False):
        from aragora.spectate.stream import SpectatorStream

        spectate_fmt = getattr(args, "spectate_format", "auto")
        extra_kwargs["spectator"] = SpectatorStream(enabled=True, format=spectate_fmt)

    extra_kwargs["auto_explain"] = True

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
            memory_config=memory_config,
            template=getattr(args, "template", None),
            mode=getattr(args, "mode", "standard") or "standard",
            verbose=getattr(args, "verbose", False),
            **extra_kwargs,
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

    # Display decision explanation if available
    if debate_result:
        explanation = getattr(debate_result, "explanation", None)
        if explanation:
            try:
                from aragora.explainability.builder import ExplanationBuilder

                summary = ExplanationBuilder().generate_summary(explanation)
                print("\nWHY THIS DECISION:")
                print("-" * 40)
                print(summary)
            except (ImportError, AttributeError, TypeError):
                pass

    # Send notification if --notify flag set
    if getattr(args, "notify", False) and debate_result:
        try:
            from aragora.notifications.service import notify_debate_completed

            asyncio.run(
                notify_debate_completed(
                    debate_id=getattr(debate_result, "debate_id", ""),
                    task=getattr(debate_result, "task", "")[:200],
                    verdict="pass"
                    if getattr(debate_result, "consensus_reached", False)
                    else "fail",
                    confidence=getattr(debate_result, "confidence", 0.0),
                    agents_used=[
                        getattr(a, "name", str(a))
                        for a in getattr(debate_result, "agents", [])[:10]
                    ],
                )
            )
            print("\nNotification sent.")
        except (ImportError, OSError) as e:
            print(f"\nNotification failed: {e}", file=sys.stderr)

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


def _print_available_templates() -> None:
    """Print available workflow templates grouped by category."""
    try:
        from aragora.workflow.templates import WORKFLOW_TEMPLATES
    except ImportError:
        print("Workflow templates not available.")
        return

    if not WORKFLOW_TEMPLATES:
        print("No templates found.")
        return

    # Group by category
    categories: dict[str, list[tuple[str, str]]] = {}
    for template_id, template in WORKFLOW_TEMPLATES.items():
        cat = template_id.split("/")[0] if "/" in template_id else "other"
        name = template.get("name", template_id)
        desc = template.get("description", "")[:60]
        categories.setdefault(cat, []).append((template_id, f"{name} - {desc}"))

    print("Available workflow templates:")
    print()
    for cat in sorted(categories):
        print(f"  {cat.upper()}:")
        for tid, desc in sorted(categories[cat]):
            print(f"    {tid:<40} {desc}")
        print()

    print(f"Total: {len(WORKFLOW_TEMPLATES)} templates")
    print('\nUsage: aragora decide --template <template-id> "your question"')


__all__ = [
    "cmd_decide",
    "cmd_plans",
    "cmd_plans_show",
    "cmd_plans_approve",
    "cmd_plans_reject",
    "cmd_plans_execute",
    "run_decide",
]

"""Swarm Commander CLI command.

Launches the full swarm lifecycle: interrogate -> spec -> dispatch -> report.

Usage:
    aragora swarm "Make the dashboard faster"
    aragora swarm "Fix tests" --skip-interrogation
    aragora swarm --spec my-spec.yaml
    aragora swarm "Add auth" --budget-limit 10
    aragora swarm "Improve UX" --dry-run
    aragora swarm "Build feature" --profile cto
    aragora swarm --from-obsidian ~/vault
    aragora swarm "Improve tests" --autonomy metrics
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4


def _resolve_swarm_action_goal(args: argparse.Namespace) -> tuple[str, str | None]:
    first = getattr(args, "swarm_action_or_goal", None)
    second = getattr(args, "swarm_goal", None)
    if first in {
        "run",
        "boss",
        "boss-loop",
        "runner",
        "status",
        "reconcile",
        "campaign",
        "integrator",
        "tranche",
    }:
        return str(first), second
    return "run", first


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _print_supervisor_run(run: dict[str, object]) -> None:
    work_orders = (
        list(run.get("work_orders", [])) if isinstance(run.get("work_orders"), list) else []
    )
    counts: dict[str, int] = {}
    for item in work_orders:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    counts_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"
    print(f"run_id={run.get('run_id', '')}")
    print(f"status={run.get('status', '')} target_branch={run.get('target_branch', '')}")
    print(f"goal={run.get('goal', '')}")
    print(f"work_orders={len(work_orders)} [{counts_text}]")


def _run_supervised_or_report(awaitable: object) -> object | None:
    try:
        return asyncio.run(awaitable)
    except ValueError as exc:
        print(f"Error: {exc}")
        return None


def _load_structured_object(source: str) -> dict[str, object]:
    if source == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(source).read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(raw) or {}
    except ImportError:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("structured input must deserialize to an object")
    return dict(payload)


def _build_boss_payload(
    run: dict[str, object],
    *,
    repo_root: Path,
    target_branch: str,
    routing: dict[str, object] | None = None,
) -> dict[str, object]:
    from aragora.swarm.reporter import build_boss_payload, build_integrator_view
    from aragora.worktree.fleet import FleetCoordinationStore, build_fleet_rows

    try:
        from aragora.nomic.dev_coordination import DevCoordinationStore
    except (ImportError, RuntimeError, OSError, ValueError):
        DevCoordinationStore = None  # type: ignore[assignment]

    worktrees = build_fleet_rows(repo_root, base_branch=target_branch, tail=0)
    store = FleetCoordinationStore(repo_root)
    claims = store.list_claims()
    merge_queue = store.list_merge_queue()
    coordination = store.status_summary()
    if DevCoordinationStore is not None:
        try:
            coordination = DevCoordinationStore(repo_root=repo_root).status_summary(
                include_integrator_artifacts=True
            )
        except (RuntimeError, OSError, ValueError):
            pass
    integrator_view = build_integrator_view(
        runs=[run],
        worktrees=worktrees,
        claims=claims,
        merge_queue=merge_queue,
        coordination=coordination,
    )
    return build_boss_payload(
        run=run,
        integrator_view=integrator_view,
        coordination=coordination,
        routing=routing,
    )


def _resolve_boss_routing() -> dict[str, object]:
    from aragora.swarm.runner_registry import LocalRunnerRegistry, authorization_context_from_env

    owner_context = authorization_context_from_env()
    return LocalRunnerRegistry().resolve_boss_routing(owner_context=owner_context).to_dict()


def _blocked_boss_payload(
    *,
    goal: str | None,
    target_branch: str,
    routing: dict[str, object],
) -> dict[str, object]:
    next_action = str(routing.get("next_action", "")).strip()
    return {
        "mode": "boss",
        "run_id": None,
        "status": "blocked",
        "goal": goal or "",
        "target_branch": target_branch,
        "work_order_counts": {},
        "lanes": [],
        "integrator_next_actions": [next_action] if next_action else [],
        "needs_human": [],
        "coordination_counts": {},
        "integrator_summary": {},
        "routing": routing,
    }


def _load_integrator_view(repo_root: Path, *, base_branch: str) -> dict[str, object]:
    from aragora.swarm.reporter import build_integrator_view
    from aragora.worktree.fleet import FleetCoordinationStore, build_fleet_rows

    try:
        from aragora.nomic.dev_coordination import DevCoordinationStore
    except (ImportError, RuntimeError, OSError, ValueError):
        DevCoordinationStore = None  # type: ignore[assignment]

    worktrees = build_fleet_rows(repo_root, base_branch=base_branch, tail=0)
    store = FleetCoordinationStore(repo_root)
    claims = store.list_claims()
    merge_queue = store.list_merge_queue()
    coordination: dict[str, object] = {}
    if DevCoordinationStore is not None:
        try:
            coordination = DevCoordinationStore(repo_root=repo_root).status_summary(
                include_integrator_artifacts=True
            )
        except (RuntimeError, OSError, ValueError):
            coordination = {}
    return build_integrator_view(
        runs=[],
        worktrees=worktrees,
        claims=claims,
        merge_queue=merge_queue,
        coordination=coordination,
    )


def _find_integrator_lane(
    view: dict[str, object],
    *,
    lane_id: str = "",
    receipt_id: str = "",
    lease_id: str = "",
    branch: str = "",
) -> dict[str, object] | None:
    lanes = [item for item in view.get("lanes", []) if isinstance(item, dict)]
    lane_id = str(lane_id or "").strip()
    receipt_id = str(receipt_id or "").strip()
    lease_id = str(lease_id or "").strip()
    branch = str(branch or "").strip()

    if lane_id:
        for lane in lanes:
            if str(lane.get("lane_id", "")).strip() == lane_id:
                return lane
        return None
    if receipt_id:
        for lane in lanes:
            if str(lane.get("receipt_id", "")).strip() == receipt_id:
                return lane
        return None
    if lease_id:
        for lane in lanes:
            if str(lane.get("lease_id", "")).strip() == lease_id:
                return lane
        return None
    if branch:
        canonical = [
            lane
            for lane in lanes
            if str(lane.get("branch", "")).strip() == branch and bool(lane.get("canonical_lane"))
        ]
        if canonical:
            return canonical[0]
        for lane in lanes:
            if str(lane.get("branch", "")).strip() == branch:
                return lane
    return None


def _render_integrator_table(view: dict[str, object]) -> None:
    summary = view.get("summary", {}) if isinstance(view.get("summary"), dict) else {}
    print(f"Swarm Integrator View ({summary.get('total_lanes', 0)} lanes)")
    print(
        "  ready={ready} blocked={blocked} review={review} stale={stale} superseded={superseded}".format(
            ready=summary.get("ready_lanes", 0),
            blocked=summary.get("blocked_lanes", 0),
            review=summary.get("review_lanes", 0),
            stale=summary.get("stale_heartbeat_lanes", 0),
            superseded=summary.get("superseded_lanes", 0),
        )
    )
    print()
    icons = {"ready": "+", "blocked": "!", "review": "?", "merged": "=", "superseded": "x"}
    lanes = [item for item in view.get("lanes", []) if isinstance(item, dict)]
    for lane in lanes:
        readiness = str(lane.get("merge_readiness", "unknown"))
        icon = icons.get(readiness, " ")
        canonical = "*" if bool(lane.get("canonical_lane")) else " "
        print(f"{canonical}[{icon}] {lane.get('title', 'untitled')}")
        print(
            "    lane_id={lane_id} branch={branch} readiness={readiness} status={status}".format(
                lane_id=lane.get("lane_id", ""),
                branch=lane.get("branch", ""),
                readiness=readiness,
                status=lane.get("status", ""),
            )
        )
        receipt_id = str(lane.get("receipt_id", "") or "").strip()
        lease_id = str(lane.get("lease_id", "") or "").strip()
        if receipt_id or lease_id:
            print(f"    receipt={receipt_id or 'none'} lease={lease_id or 'none'}")
        blockers = lane.get("blockers", [])
        if isinstance(blockers, list) and blockers:
            print(f"    blockers: {', '.join(str(item) for item in blockers)}")
        next_action = str(lane.get("next_action", "") or "").strip()
        if next_action:
            print(f"    next: {next_action}")
        print()
    for action_text in [item for item in view.get("next_actions", []) if str(item).strip()][:5]:
        print(f"next: {action_text}")


def cmd_swarm(args: argparse.Namespace) -> None:
    """Handle 'swarm' command."""
    from aragora.swarm import (
        SwarmApprovalPolicy,
        SwarmCommander,
        SwarmCommanderConfig,
        SwarmReconciler,
        SwarmSpec,
        SwarmSupervisor,
    )
    from aragora.swarm.config import (
        AutonomyLevel,
        InterrogatorConfig,
        UserProfile,
    )
    from aragora.swarm.reporter import build_integrator_view
    from aragora.worktree.fleet import (
        FleetCoordinationStore,
        build_fleet_rows,
        resolve_repo_root,
    )

    action, goal = _resolve_swarm_action_goal(args)
    spec_file = getattr(args, "spec", None)
    skip_interrogation = getattr(args, "skip_interrogation", False)
    dry_run = getattr(args, "dry_run", False)
    budget_limit = getattr(args, "budget_limit", 50.0)
    require_approval = getattr(args, "require_approval", False)
    max_parallel = getattr(args, "max_parallel", 20)
    concurrency_cap = min(max(1, int(getattr(args, "concurrency_cap", 8))), 8)
    no_loop = getattr(args, "no_loop", False)
    target_branch = getattr(args, "target_branch", "main")
    managed_dir_pattern = getattr(args, "managed_dir_pattern", ".worktrees/{agent}-auto")
    as_json = bool(getattr(args, "json", False))
    run_id = getattr(args, "run_id", None)
    refresh_scaling = bool(getattr(args, "refresh_scaling", False))
    no_dispatch = bool(getattr(args, "no_dispatch", False))
    watch = bool(getattr(args, "watch", False))
    interval_seconds = float(getattr(args, "interval_seconds", 5.0) or 5.0)
    max_ticks = getattr(args, "max_ticks", None)
    all_runs = bool(getattr(args, "all_runs", False))
    dispatch_only = bool(getattr(args, "dispatch_only", False))
    no_wait = bool(getattr(args, "no_wait", False))
    dispatch_workers = not no_dispatch
    boss_mode = action == "boss"
    boss_routing: dict[str, object] | None = None
    if dispatch_only:
        no_wait = True
    if boss_mode:
        dispatch_workers = True
        no_wait = False
        concurrency_cap = max(4, concurrency_cap)

    # Phase 2: User profile
    profile_str = getattr(args, "profile", "ceo")
    profile_map = {
        "ceo": UserProfile.CEO,
        "cto": UserProfile.CTO,
        "developer": UserProfile.DEVELOPER,
        "power-user": UserProfile.POWER_USER,
    }
    user_profile = profile_map.get(profile_str, UserProfile.CEO)

    # Phase 4: Obsidian
    from_obsidian = getattr(args, "from_obsidian", None)
    obsidian_vault = getattr(args, "obsidian_vault", None)
    no_obsidian_receipts = getattr(args, "no_obsidian_receipts", False)

    # Phase 6: Autonomy
    autonomy_str = getattr(args, "autonomy", "propose")
    autonomy_map = {
        "full-auto": AutonomyLevel.FULL_AUTO,
        "propose": AutonomyLevel.PROPOSE_APPROVE,
        "guided": AutonomyLevel.HUMAN_GUIDED,
        "metrics": AutonomyLevel.METRICS_DRIVEN,
    }
    autonomy_level = autonomy_map.get(autonomy_str, AutonomyLevel.PROPOSE_APPROVE)

    if action == "runner":
        from aragora.swarm.reporter import render_runner_registration_text
        from aragora.swarm.runner_registry import (
            CodexRunnerInspector,
            LocalRunnerRegistry,
            authorization_context_from_env,
        )

        subaction = str(goal or "inspect").strip().lower()
        if subaction not in {"inspect", "register", "heartbeat"}:
            print("Error: swarm runner action must be 'inspect', 'register', or 'heartbeat'")
            return

        inspection = CodexRunnerInspector().inspect()
        if subaction == "register":
            owner_context = authorization_context_from_env()
            payload = (
                LocalRunnerRegistry()
                .register(
                    inspection,
                    owner_context=owner_context,
                )
                .to_dict()
            )
        elif subaction == "heartbeat":
            owner_context = authorization_context_from_env()
            payload = (
                LocalRunnerRegistry()
                .heartbeat(
                    inspection,
                    owner_context=owner_context,
                )
                .to_dict()
            )
        else:
            payload = inspection.to_dict()

        payload["mode"] = "runner"
        payload["action"] = subaction
        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(render_runner_registration_text(payload))
        return

    if action == "integrator":
        import sys

        subaction = str(goal or "view").strip().lower() or "view"
        repo_root = resolve_repo_root(Path.cwd())
        base_branch = str(getattr(args, "target_branch", "main") or "main")
        view = _load_integrator_view(repo_root, base_branch=base_branch)
        readiness_filter = str(getattr(args, "readiness", None) or "").strip()
        if readiness_filter:
            filtered_view = dict(view)
            filtered_view["lanes"] = [
                item
                for item in view.get("lanes", [])
                if isinstance(item, dict)
                and str(item.get("merge_readiness", "")).strip() == readiness_filter
            ]
            view = filtered_view

        if subaction in {"view", "status"}:
            if as_json:
                print(json.dumps(view, indent=2))
            else:
                _render_integrator_table(view)
            return

        lane = _find_integrator_lane(
            view,
            lane_id=str(getattr(args, "lane_id", None) or ""),
            receipt_id=str(getattr(args, "receipt_id", None) or ""),
            lease_id=str(getattr(args, "lease_id", None) or ""),
            branch=str(getattr(args, "lane_branch", None) or ""),
        )
        if lane is None:
            print(
                "Error: integrator action requires a resolvable lane via "
                "--lane-id, --receipt-id, --lease-id, or --lane-branch",
                file=sys.stderr,
            )
            sys.exit(1)

        rationale = str(getattr(args, "rationale", "") or "").strip()
        decided_by = str(getattr(args, "decided_by", "cli-integrator") or "cli-integrator").strip()

        if subaction in {"merge", "archive"}:
            from aragora.nomic.dev_coordination import DevCoordinationStore, IntegrationDecisionType

            resolved_receipt_id = str(
                getattr(args, "receipt_id", None) or lane.get("receipt_id") or ""
            ).strip()
            resolved_lease_id = str(
                getattr(args, "lease_id", None) or lane.get("lease_id") or ""
            ).strip()
            if not resolved_receipt_id:
                print(
                    "Error: selected lane has no receipt_id; cannot record an integration decision",
                    file=sys.stderr,
                )
                sys.exit(1)

            decision_type = (
                IntegrationDecisionType.MERGE
                if subaction == "merge"
                else IntegrationDecisionType.DISCARD
            )
            decision = DevCoordinationStore(repo_root=repo_root).record_integration_decision(
                receipt_id=resolved_receipt_id,
                lease_id=resolved_lease_id or None,
                decided_by=decided_by,
                decision=decision_type,
                rationale=rationale
                or (
                    "Integrator approved lane for merge"
                    if subaction == "merge"
                    else "Integrator archived lane"
                ),
                target_branch=base_branch,
            )
            branch = str(lane.get("branch", "") or "").strip()
            if subaction == "archive" and branch:
                try:
                    from aragora.swarm.pr_registry import PullRequestRegistry

                    PullRequestRegistry().close(branch, outcome="archived")
                except (ImportError, RuntimeError, OSError, ValueError):
                    pass
            payload = {
                "lane_id": lane.get("lane_id"),
                "receipt_id": resolved_receipt_id,
                "lease_id": resolved_lease_id or None,
                "branch": branch or None,
                "decision": decision.decision,
                "decision_id": decision.decision_id,
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    "decision_id={decision_id} decision={decision} lane_id={lane_id} receipt_id={receipt_id}".format(
                        decision_id=payload["decision_id"],
                        decision=payload["decision"],
                        lane_id=payload["lane_id"],
                        receipt_id=payload["receipt_id"],
                    )
                )
            return

        if subaction == "supersede":
            from aragora.swarm.pr_registry import PullRequestRegistry

            branch = str(getattr(args, "lane_branch", None) or lane.get("branch") or "").strip()
            new_pr_url = str(getattr(args, "new_pr_url", None) or "").strip()
            if not branch or not new_pr_url:
                print(
                    "Error: integrator supersede requires a lane branch and --new-pr-url",
                    file=sys.stderr,
                )
                sys.exit(1)
            entry = PullRequestRegistry().supersede(
                branch,
                new_pr_url,
                reason=rationale or "Integrator superseded the canonical PR",
            )
            if entry is None:
                print(f"Error: branch not found in PR registry: {branch}", file=sys.stderr)
                sys.exit(1)
            payload = {
                "branch": branch,
                "new_pr_url": new_pr_url,
                "status": entry.status,
                "superseded_count": len(entry.superseded),
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    "branch={branch} superseded_count={count} new_pr={url}".format(
                        branch=branch,
                        count=payload["superseded_count"],
                        url=new_pr_url,
                    )
                )
            return

        print(
            "Error: swarm integrator action must be one of view, status, merge, archive, or supersede",
            file=sys.stderr,
        )
        sys.exit(1)

    if action == "boss-loop":
        from aragora.swarm.boss_loop import BossLoop, BossLoopConfig

        boss_loop_config = BossLoopConfig(
            max_iterations=int(getattr(args, "max_ticks", None) or 50),
            iteration_interval_seconds=float(getattr(args, "interval_seconds", 30.0) or 30.0),
            freshness_ttl_seconds=float(getattr(args, "freshness_ttl", 3600.0) or 3600.0),
            repo=getattr(args, "boss_repo", None),
            label_filter=getattr(args, "boss_label_filter", None),
            issue_number=getattr(args, "boss_issue_number", None),
            target_branch=target_branch,
            budget_limit_usd=budget_limit,
            max_consecutive_failures=int(getattr(args, "max_consecutive_failures", 3) or 3),
            require_validation_contract=not bool(
                getattr(args, "allow_missing_validation_contract", False)
            ),
            dispatch_enabled=not no_dispatch,
        )
        loop = BossLoop(config=boss_loop_config)

        def _on_status(status: object) -> None:
            if as_json:
                return  # JSON output is emitted at the end
            status_dict = status.to_dict() if hasattr(status, "to_dict") else {}
            iteration = status_dict.get("iteration", "?")
            worker = status_dict.get("worker_status", "?")
            issue = status_dict.get("selected_issue")
            issue_text = (
                f"#{issue.get('number', '?')} {issue.get('title', '')[:60]}"
                if isinstance(issue, dict)
                else "none"
            )
            stop = status_dict.get("stop_reason")
            print(
                f"[iter {iteration}] worker={worker} issue={issue_text}"
                + (f" stop={stop}" if stop else "")
            )
            for action_text in status_dict.get("next_actions", [])[:2]:
                print(f"  next: {action_text}")

        result = asyncio.run(loop.run(on_status=_on_status))
        if as_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"\nBoss loop finished: {result.stop_reason}")
            print(
                f"iterations={result.iterations_completed} "
                f"attempted={len(result.issues_attempted)} "
                f"completed={len(result.issues_completed)} "
                f"failed={len(result.issues_failed)} "
                f"elapsed={result.total_elapsed_seconds:.1f}s"
            )
            for reason in result.needs_human_reasons[:3]:
                print(f"  needs_human: {reason}")
            for action_text in result.next_actions[:3]:
                print(f"  next: {action_text}")
        return

    if action == "campaign":
        from aragora.swarm.campaign import (
            CampaignExecutor,
            CampaignPlanner,
            DEFAULT_CAMPAIGN_MANIFEST,
            load_campaign_manifest,
            locked_manifest_path,
            save_campaign_manifest,
        )

        subaction = str(goal or "status").strip().lower()
        manifest_path = Path(getattr(args, "manifest", None) or DEFAULT_CAMPAIGN_MANIFEST).resolve()
        output_path = Path(getattr(args, "output", None) or manifest_path).resolve()

        def _campaign_input_count() -> int:
            return sum(
                1
                for value in (
                    getattr(args, "source_file", None),
                    getattr(args, "issue_list", None),
                    getattr(args, "github_query", None),
                )
                if value
            )

        def _campaign_planner(parallel_default: int = 1):
            return CampaignPlanner(
                repo_root=Path.cwd(),
                planner_model=str(getattr(args, "planner_model", "claude") or "claude"),
                planner_strategy=str(getattr(args, "planner_strategy", "heuristic") or "heuristic"),
                worker_model=str(getattr(args, "worker_model", "codex") or "codex"),
                review_model=str(getattr(args, "review_model", "claude") or "claude"),
                enforce_cross_model_review=not bool(
                    getattr(args, "allow_same_model_review", False)
                ),
                budget_limit_usd=float(getattr(args, "budget_limit", 50.0) or 50.0),
                max_parallel_ready_projects=int(
                    getattr(args, "max_parallel_ready_projects", parallel_default)
                    or parallel_default
                ),
                experiment_id=str(getattr(args, "experiment_id", "")).strip() or None,
                experiment_label=str(getattr(args, "experiment_label", "")).strip() or None,
            )

        def _plan_campaign(planner):
            source_file = getattr(args, "source_file", None)
            issue_list = getattr(args, "issue_list", None)
            github_query = getattr(args, "github_query", None)
            if source_file:
                return planner.plan_from_source_file(Path(source_file).resolve())
            if issue_list:
                issue_numbers = [
                    int(item.strip()) for item in str(issue_list).split(",") if item.strip()
                ]
                return planner.plan_from_issue_list(
                    issue_numbers,
                    repo=getattr(args, "boss_repo", None),
                )
            if github_query:
                return planner.plan_from_github_query(
                    str(github_query),
                    repo=getattr(args, "boss_repo", None),
                )
            raise ValueError(
                "campaign plan requires exactly one of --source-file, --issue-list, or --github-query"
            )

        if subaction == "plan":
            if _campaign_input_count() != 1:
                raise ValueError(
                    "campaign plan requires exactly one of --source-file, --issue-list, or --github-query"
                )
            planner = _campaign_planner(parallel_default=1)
            manifest = _plan_campaign(planner)
            with locked_manifest_path(output_path):
                save_campaign_manifest(output_path, manifest)
            payload = {
                "mode": "campaign-plan",
                "manifest_path": str(output_path),
                **manifest.to_dict(),
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"campaign_id={manifest.campaign_id}")
                print(f"manifest={output_path}")
                print(
                    f"projects={len(manifest.projects)} budget=${manifest.budget_limit_usd:.2f} "
                    f"worker={manifest.worker_model} review={manifest.review_model}"
                )
                for finding in manifest.planning_findings[:5]:
                    print(f"  finding: {finding}")
            return

        if subaction == "run":
            # Unified pipeline: plan once into a canonical manifest, then execute exactly one iteration.
            source_count = _campaign_input_count()
            run_manifest_path = manifest_path
            invocation_mode = "resumed"
            if manifest_path.exists():
                if source_count > 0:
                    raise ValueError(
                        "campaign run: cannot supply --source-file, --issue-list, or "
                        "--github-query when resuming from an existing manifest"
                    )
                if not as_json:
                    print(f"Resuming from existing manifest: {manifest_path}")
            else:
                if source_count == 0:
                    raise ValueError(
                        "campaign run requires an existing manifest or one of "
                        "--source-file, --issue-list, --github-query"
                    )
                if source_count != 1:
                    raise ValueError(
                        "campaign run requires exactly one of --source-file, --issue-list, or "
                        "--github-query when the manifest does not exist"
                    )
                planner = _campaign_planner(parallel_default=1)
                manifest = _plan_campaign(planner)
                run_manifest_path = output_path
                with locked_manifest_path(run_manifest_path):
                    save_campaign_manifest(output_path, manifest)
                invocation_mode = "planned_then_executed"
                if not as_json:
                    print(f"Planned {len(manifest.projects)} projects → {run_manifest_path}")
            executor = CampaignExecutor(
                manifest_path=run_manifest_path,
                repo_root=Path.cwd(),
                target_branch=target_branch,
            )
            payload = {
                "mode": "campaign-run",
                "invocation_mode": invocation_mode,
                "manifest_path": str(run_manifest_path),
                **asyncio.run(executor.execute_once()),
            }
            with locked_manifest_path(run_manifest_path):
                manifest = load_campaign_manifest(run_manifest_path)
                payload["campaign_id"] = manifest.campaign_id
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                stop = payload.get("stop_reason", "")
                dispatched = payload.get("dispatched_projects", [])
                print(
                    f"campaign_id={payload.get('campaign_id', '')} "
                    f"manifest={run_manifest_path} "
                    f"invocation_mode={invocation_mode} "
                    f"stop_reason={stop} dispatched={len(dispatched)}"
                )
                for item in dispatched:
                    if isinstance(item, dict):
                        print(
                            f"  {item.get('project_id')} status={item.get('status')} "
                            f"outcome={item.get('outcome')}"
                        )
                    elif isinstance(item, str):
                        print(f"  {item}")
            return

        executor = CampaignExecutor(
            manifest_path=manifest_path,
            repo_root=Path.cwd(),
            target_branch=target_branch,
        )
        if subaction == "execute":
            payload = asyncio.run(executor.execute_once())
        elif subaction == "status":
            payload = executor.status()
        elif subaction == "review":
            target = str(getattr(args, "swarm_campaign_target", None) or "").strip()
            if not target:
                raise ValueError("campaign review requires a project id as the third argument")
            payload = asyncio.run(executor.review_project(target))
        elif subaction == "sync-issues":
            payload = executor.sync_issue_plan()
        else:
            raise ValueError(
                "campaign action must be one of: plan, run, execute, status, review, sync-issues"
            )

        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            if subaction == "status":
                print(
                    f"campaign_id={payload.get('campaign_id', '')} "
                    f"stop_reason={payload.get('stop_reason', '')}"
                )
                counts = payload.get("counts", {})
                if isinstance(counts, dict):
                    counts_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                    print(f"counts={counts_text}")
                for project in payload.get("projects", [])[:10]:
                    if not isinstance(project, dict):
                        continue
                    print(
                        f"{project.get('project_id')} status={project.get('status')} "
                        f"review={project.get('review_status')} title={project.get('title', '')}"
                    )
            else:
                print(json.dumps(payload, indent=2))
        return

    if action == "tranche":
        import os

        from aragora.nomic.dev_coordination import DevCoordinationStore
        from aragora.ralph.github_control import GitHubControl
        from aragora.swarm.pr_registry import PullRequestRegistry
        from aragora.swarm.tranche import (
            TrancheArtifactStore,
            TrancheExecutor,
            TrancheInspector,
            TranchePlanner,
            load_tranche_manifest,
            render_tranche_inspection_text,
        )
        from aragora.swarm.tranche_integrate import (
            assess_lane_integration,
            classify_check_results,
            discover_lane_pr,
            record_lane_integration,
            register_pr,
        )
        from aragora.swarm.tranche_review import review_lane, select_review_tier
        from aragora.swarm.tranche_submit import submit_intake_bundle
        from aragora.swarm.tranche_watch import (
            claim_driver,
            list_tranche_states,
            load_tranche_run_state,
            release_driver,
            run_state_path_for_manifest,
            watch_loop,
        )

        subaction = str(goal or "inspect").strip().lower() or "inspect"
        repo_root = resolve_repo_root(Path.cwd())
        if subaction == "submit":
            intake_arg = str(getattr(args, "intake", "") or "").strip()
            if not intake_arg:
                raise ValueError("tranche submit requires --intake <path|->")
            intake_path: Path | None = None
            if intake_arg != "-":
                intake_path = Path(intake_arg).resolve()
                if not intake_path.exists():
                    raise ValueError(f"intake bundle not found: {intake_path}")
            bundle = _load_structured_object(intake_arg)
            payload = submit_intake_bundle(
                bundle,
                repo_root=repo_root,
                autonomy_mode=_optional_text(getattr(args, "autonomy", None)),
            )
            payload["mode"] = "tranche-submit"
            payload["action"] = subaction
            if intake_path is not None:
                payload["intake_path"] = str(intake_path)
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(json.dumps(payload, indent=2))
            return
        if subaction == "list":
            items = list_tranche_states(repo_root)
            payload = {
                "mode": "tranche-list",
                "action": subaction,
                "count": len(items),
                "items": items,
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(json.dumps(payload, indent=2))
            return
        if subaction == "plan":
            prompt_arg = str(getattr(args, "from_prompts", "") or "").strip()
            if not prompt_arg:
                raise ValueError("tranche plan requires --from-prompts <path>")
            prompt_path = Path(prompt_arg).resolve()
            if not prompt_path.exists():
                raise ValueError(f"prompt bundle not found: {prompt_path}")
            manifest_arg = str(getattr(args, "manifest", "") or "").strip()
            output_arg = str(getattr(args, "output", "") or "").strip()
            output_path: Path | None = None
            if output_arg:
                output_path = Path(output_arg).resolve()
            elif manifest_arg and manifest_arg != ".aragora/campaign_manifest.yaml":
                output_path = Path(manifest_arg).resolve()
            planner = TranchePlanner(repo_root=repo_root)
            manifest, saved_path = planner.plan_from_prompt_bundle(
                prompt_path,
                output_path=output_path,
            )
            payload = {
                "mode": "tranche-plan",
                "action": subaction,
                "manifest_id": manifest.manifest_id,
                "manifest_path": str(saved_path),
                "lane_count": len(manifest.lanes),
                "reference_groups": sorted(manifest.references),
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"manifest_id={manifest.manifest_id}")
                print(f"manifest_path={saved_path}")
                print(f"lanes={len(manifest.lanes)}")
            return

        manifest_arg = str(getattr(args, "manifest", "") or "").strip()
        if not manifest_arg:
            raise ValueError(f"tranche {subaction} requires --manifest <path>")
        manifest_path = Path(manifest_arg).resolve()
        if not manifest_path.exists():
            raise ValueError(f"tranche manifest not found: {manifest_path}")
        manifest = load_tranche_manifest(manifest_path)

        if subaction == "inspect":
            payload = TrancheInspector(repo_root=repo_root).inspect(manifest)
            payload["action"] = subaction
            payload["manifest_path"] = str(manifest_path)
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(render_tranche_inspection_text(payload))
            return

        if subaction == "watch":
            state_path = run_state_path_for_manifest(manifest_path)
            state = load_tranche_run_state(manifest_path)
            artifact_store = TrancheArtifactStore(repo_root=repo_root)
            driver_mode = bool(getattr(args, "driver", False))
            session_id = str(
                getattr(args, "owner_session_id", None) or f"cli-watch-{os.getpid()}"
            ).strip()
            supervisor = None
            github = None
            registry = None

            async def _watch_review_fn(*, manifest, lane_id, artifact):
                nonlocal supervisor
                from aragora.swarm.supervisor import SwarmSupervisor

                if artifact is None:
                    return {
                        "status": "blocked_nonreviewable",
                        "findings": ["Missing tranche artifact."],
                    }
                run_id = str(getattr(artifact, "run_id", None) or "").strip()
                if not run_id:
                    return {
                        "status": "blocked_nonreviewable",
                        "findings": ["Artifact has no run_id."],
                    }
                if supervisor is None:
                    supervisor = SwarmSupervisor(repo_root=repo_root)
                try:
                    run_dict = supervisor.refresh_run(run_id).to_dict()
                except Exception:
                    record = supervisor.store.get_supervisor_run(run_id)
                    if not isinstance(record, dict):
                        return {
                            "status": "blocked_nonreviewable",
                            "findings": [f"Supervisor run {run_id} is not available."],
                        }
                    run_dict = dict(record)
                lane = manifest.lane(lane_id)
                tier = select_review_tier(
                    write_scope=list(getattr(lane, "allowed_write_scope", [])),
                    diff_lines=int(getattr(artifact, "metadata", {}).get("diff_lines", 0) or 0),
                    verification_passed=bool(getattr(artifact, "commands", [])),
                    risk_tolerance=str(
                        getattr(artifact, "metadata", {}).get("risk_tolerance", "") or ""
                    ).strip()
                    or None,
                )
                return await review_lane(
                    manifest=manifest,
                    lane_id=lane_id,
                    artifact=artifact,
                    run_dict=run_dict,
                    tier=tier,
                    repo_root=repo_root,
                )

            async def _watch_integrate_fn(*, manifest, lane_id, artifact, approve):
                nonlocal github, registry
                if artifact is None:
                    return {"recommendation": "needs_human", "executed": False}
                if github is None:
                    github = GitHubControl(repo_root=repo_root)
                if registry is None:
                    registry = PullRequestRegistry()
                metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
                review_payload = metadata.get("review", {})
                if isinstance(review_payload, dict):
                    review_status = str(review_payload.get("status", "") or "").strip() or "pending"
                else:
                    review_status = "pending"
                if review_status == "pending":
                    artifact_status = str(getattr(artifact, "status", "") or "").strip()
                    if artifact_status == "review_passed":
                        review_status = "passed"
                    elif artifact_status == "changes_requested":
                        review_status = "changes_requested"
                    elif artifact_status == "review_blocked":
                        review_status = "blocked_nonreviewable"

                pr_url = discover_lane_pr(artifact, github=github, repo_root=repo_root)
                branch = str(
                    metadata.get("branch")
                    or (
                        metadata.get("deliverable", {}).get("branch")
                        if isinstance(metadata.get("deliverable"), dict)
                        else ""
                    )
                    or ""
                ).strip()
                if pr_url and branch:
                    register_pr(pr_url, branch, registry)

                checks = "checks_pending"
                merge_result = None
                if pr_url:
                    snapshot = github.fetch_gate_snapshot(pr_url)
                    checks = classify_check_results(
                        list(getattr(snapshot, "required_checks", []))
                        + list(getattr(snapshot, "advisory_checks", []))
                    )
                assessment = assess_lane_integration(
                    artifact=artifact,
                    checks=checks,
                    review_status=review_status,
                    merge_policy=str(metadata.get("merge_policy", "confirm") or "confirm"),
                    autonomy_mode=str(state.autonomy_mode or "adaptive"),
                    approve=bool(approve),
                )
                if approve and assessment.get("recommendation") == "merge":
                    coord_store = DevCoordinationStore(repo_root=repo_root)
                    await record_lane_integration(
                        artifact=artifact,
                        decision="merge",
                        rationale="Tranche watch approved merge after green checks and review.",
                        decided_by="tranche-watch",
                        store=coord_store,
                        target_branch=str(getattr(args, "target_branch", "main") or "main"),
                    )
                    if pr_url and assessment.get("executed"):
                        merge_call = github.merge_pr(
                            pr_url,
                            required_checks_green=checks == "checks_passed",
                            allow_admin=False,
                        )
                        merge_result = (
                            merge_call.to_dict() if hasattr(merge_call, "to_dict") else None
                        )
                        if isinstance(merge_result, dict):
                            assessment["executed"] = bool(merge_result.get("merged", False))
                if merge_result is not None:
                    assessment["merge_result"] = merge_result
                return assessment

            if driver_mode:
                state = claim_driver(state, session_id=session_id)
                state.save(state_path)
            final_state = asyncio.run(
                watch_loop(
                    state,
                    manifest=manifest,
                    interval_seconds=interval_seconds,
                    max_ticks=max_ticks,
                    state_path=state_path,
                    artifact_store=artifact_store,
                    repo_root=repo_root,
                    review_fn=_watch_review_fn if driver_mode else None,
                    integrate_fn=_watch_integrate_fn if driver_mode else None,
                )
            )
            if driver_mode:
                final_state = release_driver(final_state, session_id=session_id)
                final_state.save(state_path)
            payload = {
                "mode": "tranche-watch",
                "action": subaction,
                "manifest_id": manifest.manifest_id,
                "manifest_path": str(manifest_path),
                "driver": driver_mode,
                **final_state.to_dict(),
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(json.dumps(payload, indent=2))
            return

        if subaction == "design-review":
            from aragora.swarm.tranche_design_review import (
                DesignReviewRecord,
                run_design_review,
                save_design_review,
            )

            inspection = TrancheInspector(repo_root=repo_root).inspect(manifest)
            normalized_path = manifest_path.with_name("normalized_bundle.yaml")
            if normalized_path.exists():
                normalized_bundle = _load_structured_object(str(normalized_path))
            else:
                normalized_bundle = {
                    "manifest_id": getattr(manifest, "manifest_id", ""),
                    "objective": getattr(manifest, "objective", ""),
                    "lanes": [
                        lane.to_dict()
                        for lane in getattr(manifest, "lanes", [])
                        if hasattr(lane, "to_dict")
                    ],
                }
            payload = asyncio.run(
                run_design_review(
                    manifest=manifest,
                    normalized_bundle=normalized_bundle,
                    inspection=inspection,
                    max_rounds=int(getattr(args, "rounds", 2) or 2),
                )
            )
            record_payload = payload.get("record")
            if isinstance(record_payload, dict):
                save_design_review(
                    manifest_path.with_name("design_review.yaml"),
                    DesignReviewRecord.from_dict(record_payload),
                )
            payload["action"] = subaction
            payload["manifest_path"] = str(manifest_path)
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(json.dumps(payload, indent=2))
            return

        if subaction == "review":
            from aragora.swarm.supervisor import SwarmSupervisor

            artifact_store = TrancheArtifactStore(repo_root=repo_root)
            lane_id = str(getattr(args, "lane_id", "") or "").strip()
            all_completed = bool(getattr(args, "all_completed", False))
            if lane_id:
                artifact = artifact_store.load(manifest.manifest_id, lane_id)
                selected_artifacts = [artifact] if artifact is not None else []
            elif all_completed:
                selected_artifacts = [
                    item
                    for item in artifact_store.list(manifest.manifest_id)
                    if str(item.status).strip()
                    in {"completed", "review_passed", "changes_requested", "review_blocked"}
                ]
            else:
                raise ValueError("tranche review requires --lane-id <id> or --all-completed")
            if not selected_artifacts:
                raise ValueError("No matching tranche artifacts found for review.")
            supervisor = SwarmSupervisor(repo_root=repo_root)
            results: list[dict[str, object]] = []
            for artifact in selected_artifacts:
                run_id = str(getattr(artifact, "run_id", None) or "").strip()
                if not run_id:
                    raise ValueError(f"Artifact {artifact.lane_id} has no run_id.")
                try:
                    run_dict = supervisor.refresh_run(run_id).to_dict()
                except Exception:
                    record = supervisor.store.get_supervisor_run(run_id)
                    if not isinstance(record, dict):
                        raise ValueError(f"Supervisor run {run_id} is not available.") from None
                    run_dict = dict(record)
                tier_arg = str(getattr(args, "tier", "auto") or "auto").strip()
                if tier_arg == "auto":
                    lane = manifest.lane(artifact.lane_id)
                    tier = select_review_tier(
                        write_scope=list(getattr(lane, "allowed_write_scope", [])),
                        diff_lines=int(getattr(artifact, "metadata", {}).get("diff_lines", 0) or 0),
                        verification_passed=bool(getattr(artifact, "commands", [])),
                        risk_tolerance=str(
                            getattr(artifact, "metadata", {}).get("risk_tolerance", "") or ""
                        ).strip()
                        or None,
                    )
                else:
                    tier = int(tier_arg)
                review_payload = asyncio.run(
                    review_lane(
                        manifest=manifest,
                        lane_id=artifact.lane_id,
                        artifact=artifact,
                        run_dict=run_dict,
                        tier=tier,
                        repo_root=repo_root,
                    )
                )
                results.append({"lane_id": artifact.lane_id, **review_payload})
            payload = {
                "mode": "tranche-review",
                "action": subaction,
                "manifest_id": manifest.manifest_id,
                "manifest_path": str(manifest_path),
                "results": results,
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(json.dumps(payload, indent=2))
            return

        if subaction == "integrate":
            artifact_store = TrancheArtifactStore(repo_root=repo_root)
            lane_id = str(getattr(args, "lane_id", "") or "").strip()
            all_mergeable = bool(getattr(args, "all_mergeable", False))
            approve = bool(getattr(args, "approve", False))
            if lane_id:
                artifact = artifact_store.load(manifest.manifest_id, lane_id)
                selected_artifacts = [artifact] if artifact is not None else []
            elif all_mergeable:
                selected_artifacts = [
                    item
                    for item in artifact_store.list(manifest.manifest_id)
                    if str(item.status).strip() in {"review_passed", "completed"}
                ]
            else:
                raise ValueError("tranche integrate requires --lane-id <id> or --all-mergeable")
            if not selected_artifacts:
                raise ValueError("No matching tranche artifacts found for integrate.")

            github = GitHubControl(repo_root=repo_root)
            registry = PullRequestRegistry()
            store = DevCoordinationStore(repo_root=repo_root) if approve else None
            results: list[dict[str, object]] = []
            for artifact in selected_artifacts:
                metadata = getattr(artifact, "metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                review_payload = metadata.get("review", {})
                if isinstance(review_payload, dict):
                    review_status = str(review_payload.get("status", "") or "").strip() or "pending"
                else:
                    review_status = "pending"
                if review_status == "pending":
                    status_text = str(getattr(artifact, "status", "") or "").strip()
                    if status_text == "review_passed":
                        review_status = "passed"
                    elif status_text == "changes_requested":
                        review_status = "changes_requested"
                    elif status_text == "review_blocked":
                        review_status = "blocked_nonreviewable"

                pr_url = discover_lane_pr(artifact, github=github, repo_root=repo_root)
                branch = str(
                    metadata.get("branch")
                    or (
                        metadata.get("deliverable", {}).get("branch")
                        if isinstance(metadata.get("deliverable"), dict)
                        else ""
                    )
                    or ""
                ).strip()
                if pr_url and branch:
                    register_pr(pr_url, branch, registry)

                gate_payload: dict[str, object] | None = None
                checks = "checks_pending"
                merge_result: dict[str, object] | None = None
                if pr_url:
                    snapshot = github.fetch_gate_snapshot(pr_url)
                    gate_payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else None
                    checks = classify_check_results(
                        list(getattr(snapshot, "required_checks", []))
                        + list(getattr(snapshot, "advisory_checks", []))
                    )

                assessment = assess_lane_integration(
                    artifact=artifact,
                    checks=checks,
                    review_status=review_status,
                    merge_policy=str(metadata.get("merge_policy", "confirm") or "confirm"),
                    autonomy_mode=str(getattr(args, "autonomy", "adaptive") or "adaptive"),
                    approve=approve,
                )
                if approve and assessment.get("recommendation") == "merge":
                    awaitable = record_lane_integration(
                        artifact=artifact,
                        decision="merge",
                        rationale=str(
                            getattr(args, "rationale", None)
                            or "Tranche integrate approved merge after green checks and review."
                        ),
                        decided_by=str(getattr(args, "decided_by", None) or "tranche-integrate"),
                        store=store,
                        target_branch=str(getattr(args, "target_branch", "main") or "main"),
                    )
                    asyncio.run(awaitable)
                    if pr_url and assessment.get("executed"):
                        merge_call = github.merge_pr(
                            pr_url,
                            required_checks_green=checks == "checks_passed",
                            allow_admin=False,
                        )
                        merge_result = (
                            merge_call.to_dict() if hasattr(merge_call, "to_dict") else None
                        )
                        if isinstance(merge_result, dict):
                            assessment["executed"] = bool(merge_result.get("merged", False))

                results.append(
                    {
                        "lane_id": artifact.lane_id,
                        "status": getattr(artifact, "status", None),
                        "pr_url": pr_url,
                        "checks": checks,
                        "review_status": review_status,
                        "gate": gate_payload,
                        "merge_result": merge_result,
                        **assessment,
                    }
                )

            payload = {
                "mode": "tranche-integrate",
                "action": subaction,
                "manifest_id": manifest.manifest_id,
                "manifest_path": str(manifest_path),
                "approve": approve,
                "results": results,
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(json.dumps(payload, indent=2))
            return

        executor = TrancheExecutor(repo_root=repo_root)
        lane_id = str(getattr(args, "lane_id", "") or "").strip()
        all_ready = bool(getattr(args, "all_ready", False))
        owner_agent = _optional_text(getattr(args, "owner_agent", None))
        owner_session_id = _optional_text(getattr(args, "owner_session_id", None))
        if subaction == "prepare":
            payload = executor.prepare(
                manifest,
                lane_id=lane_id,
                all_ready=all_ready,
                owner_agent=owner_agent,
                owner_session_id=owner_session_id,
                base_branch=str(getattr(args, "target_branch", "main") or "main"),
            )
        elif subaction == "run":
            payload = asyncio.run(
                executor.run(
                    manifest,
                    lane_id=lane_id,
                    all_ready=all_ready,
                    owner_agent=owner_agent,
                    owner_session_id=owner_session_id,
                    target_branch=str(getattr(args, "target_branch", "main") or "main"),
                    max_ticks=int(getattr(args, "max_ticks", 360) or 360),
                    wait_for_completion=not bool(getattr(args, "no_wait", False)),
                    skip_review=bool(getattr(args, "skip_review", False)),
                )
            )
        else:
            raise ValueError(
                "tranche action must be one of: submit, plan, inspect, watch, list, design-review, review, integrate, prepare, run"
            )
        payload["action"] = subaction
        payload["manifest_path"] = str(manifest_path)
        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return

    if boss_mode:
        boss_routing = _resolve_boss_routing()
        blocked_reason = boss_routing.get("blocked_reason")
        if isinstance(blocked_reason, str) and blocked_reason.strip():
            from aragora.swarm.reporter import render_boss_text

            payload = _blocked_boss_payload(
                goal=goal,
                target_branch=target_branch,
                routing=boss_routing,
            )
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(render_boss_text(payload))
            return
    if action == "status":
        repo_root = resolve_repo_root(Path.cwd())
        supervisor = SwarmSupervisor(repo_root=repo_root)
        payload = supervisor.status_summary(
            run_id=run_id,
            limit=int(getattr(args, "status_limit", 20)),
            refresh_scaling=refresh_scaling,
        )
        base_branch = str(getattr(args, "target_branch", "main") or "main")
        worktrees = build_fleet_rows(repo_root, base_branch=base_branch, tail=0)
        store = FleetCoordinationStore(repo_root)
        claims = store.list_claims()
        merge_queue = store.list_merge_queue()
        payload["integrator_view"] = build_integrator_view(
            runs=payload.get("runs", []),
            worktrees=worktrees,
            claims=claims,
            merge_queue=merge_queue,
            coordination=payload.get("coordination", {}),
        )
        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                "runs={runs} queued={queued} leased={leased} completed={completed}".format(
                    runs=payload["counts"].get("runs", 0),
                    queued=payload["counts"].get("queued_work_orders", 0),
                    leased=payload["counts"].get("leased_work_orders", 0),
                    completed=payload["counts"].get("completed_work_orders", 0),
                )
            )
            integrator_summary = payload["integrator_view"].get("summary", {})
            print(
                "integrator ready={ready} review={review} blocked={blocked} stale={stale} "
                "collisions={collisions} missing_receipts={missing} superseded={superseded}".format(
                    ready=integrator_summary.get("ready_lanes", 0),
                    review=integrator_summary.get("review_lanes", 0),
                    blocked=integrator_summary.get("blocked_lanes", 0),
                    stale=integrator_summary.get("stale_heartbeat_lanes", 0),
                    collisions=integrator_summary.get("collision_lanes", 0),
                    missing=integrator_summary.get("missing_receipt_lanes", 0),
                    superseded=integrator_summary.get("superseded_lanes", 0),
                )
            )
            for action_text in payload["integrator_view"].get("next_actions", [])[:3]:
                print(f"next: {action_text}")
            for run in payload.get("runs", []):
                if isinstance(run, dict):
                    print("---")
                    _print_supervisor_run(run)
        return

    if action == "reconcile":
        reconciler = SwarmReconciler(repo_root=Path.cwd())
        if all_runs:
            runs = asyncio.run(
                reconciler.tick_open_runs(limit=int(getattr(args, "status_limit", 20)))
            )
            payload = {"runs": [run.to_dict() for run in runs], "count": len(runs)}
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"runs={payload['count']}")
                for run in payload["runs"]:
                    print("---")
                    _print_supervisor_run(run)
            return
        if not run_id:
            print("Error: provide --run-id or --all-runs for 'reconcile'")
            return
        run = asyncio.run(
            reconciler.watch_run(
                run_id,
                interval_seconds=interval_seconds,
                max_ticks=max_ticks,
            )
            if watch
            else reconciler.tick_run(run_id)
        )
        if as_json:
            print(json.dumps(run.to_dict(), indent=2))
        else:
            _print_supervisor_run(run.to_dict())
        return

    if not goal and not spec_file and not from_obsidian:
        print("Error: provide a goal or --spec file (or --from-obsidian vault)")
        print('Usage: aragora swarm run "your goal here"')
        return

    config = SwarmCommanderConfig(
        interrogator=InterrogatorConfig(user_profile=user_profile),
        budget_limit_usd=budget_limit,
        require_approval=require_approval,
        max_parallel_tasks=max_parallel,
        iterative_mode=not no_loop,
        user_profile=user_profile,
        obsidian_vault_path=obsidian_vault or from_obsidian,
        obsidian_write_receipts=not no_obsidian_receipts,
        autonomy_level=autonomy_level,
    )
    commander = SwarmCommander(config=config)
    approval_policy = SwarmApprovalPolicy(
        require_merge_approval=True,
        require_external_action_approval=True,
    )

    # Phase 4: Load goals from Obsidian
    if from_obsidian and not goal:
        goals = asyncio.run(commander._load_from_obsidian(from_obsidian))
        if goals:
            goal = goals[0]  # Use first tagged note as goal
            print(f"\nLoaded goal from Obsidian: {goal[:100]}...")
        else:
            print("No #swarm tagged notes found in Obsidian vault")
            return

    if spec_file:
        spec_path = Path(spec_file)
        if not spec_path.exists():
            print(f"Error: spec file not found: {spec_file}")
            return
        spec = SwarmSpec.from_yaml(spec_path.read_text())
        print(f"\nLoaded spec from {spec_file}")
        print(spec.summary())
        run = _run_supervised_or_report(
            commander.run_supervised_from_spec(
                spec,
                repo_path=Path.cwd(),
                target_branch=target_branch,
                max_concurrency=concurrency_cap,
                managed_dir_pattern=managed_dir_pattern,
                approval_policy=approval_policy,
                dispatch=dispatch_workers,
                wait=not no_wait,
                interval_seconds=interval_seconds,
                max_ticks=max_ticks,
            )
        )
        if run is None:
            return
        run_payload = run.to_dict()
        if boss_mode:
            boss_payload = _build_boss_payload(
                run_payload,
                repo_root=Path.cwd(),
                target_branch=target_branch,
                routing=boss_routing,
            )
            if as_json:
                print(json.dumps(boss_payload, indent=2))
            else:
                from aragora.swarm.reporter import render_boss_text

                print(render_boss_text(boss_payload))
            return
        if as_json:
            print(json.dumps(run_payload, indent=2))
        else:
            _print_supervisor_run(run_payload)
    elif dry_run:
        if skip_interrogation:
            spec = SwarmSpec(
                id=str(uuid4()),
                created_at=datetime.now(timezone.utc),
                raw_goal=goal,
                refined_goal=goal,
                budget_limit_usd=budget_limit,
                requires_approval=require_approval,
                interrogation_turns=0,
                user_expertise="developer",
            )
            print("\n[DRY RUN] Skipping interrogation and building a direct spec.\n")
            print(spec.to_json(indent=2))
        else:
            spec = asyncio.run(commander.dry_run(goal))
        save_path = getattr(args, "save_spec", None)
        if save_path:
            Path(save_path).write_text(spec.to_yaml())
            print(f"\nSpec saved to {save_path}")
    elif skip_interrogation:
        spec = SwarmSpec.from_direct_goal(
            goal,
            budget_limit_usd=budget_limit,
            requires_approval=require_approval,
            user_expertise="developer",
        )
        print("\nSkipping interrogation (developer mode)")
        print(spec.summary())
        if not spec.is_dispatch_bounded():
            print(f"Error: {spec.dispatch_gate_reason()}")
            return
        run = _run_supervised_or_report(
            commander.run_supervised_from_spec(
                spec,
                repo_path=Path.cwd(),
                target_branch=target_branch,
                max_concurrency=concurrency_cap,
                managed_dir_pattern=managed_dir_pattern,
                approval_policy=approval_policy,
                dispatch=dispatch_workers,
                wait=not no_wait,
                interval_seconds=interval_seconds,
                max_ticks=max_ticks,
            )
        )
        if run is None:
            return
        run_payload = run.to_dict()
        if boss_mode:
            boss_payload = _build_boss_payload(
                run_payload,
                repo_root=Path.cwd(),
                target_branch=target_branch,
                routing=boss_routing,
            )
            if as_json:
                print(json.dumps(boss_payload, indent=2))
            else:
                from aragora.swarm.reporter import render_boss_text

                print(render_boss_text(boss_payload))
            return
        if as_json:
            print(json.dumps(run_payload, indent=2))
        else:
            _print_supervisor_run(run_payload)
    else:
        run = _run_supervised_or_report(
            commander.run_supervised(
                goal,
                repo_path=Path.cwd(),
                target_branch=target_branch,
                max_concurrency=concurrency_cap,
                managed_dir_pattern=managed_dir_pattern,
                approval_policy=approval_policy,
                dispatch=dispatch_workers,
                wait=not no_wait,
                interval_seconds=interval_seconds,
                max_ticks=max_ticks,
            )
        )
        if run is None:
            return
        run_payload = run.to_dict()
        if boss_mode:
            boss_payload = _build_boss_payload(
                run_payload,
                repo_root=Path.cwd(),
                target_branch=target_branch,
                routing=boss_routing,
            )
            if as_json:
                print(json.dumps(boss_payload, indent=2))
            else:
                from aragora.swarm.reporter import render_boss_text

                print(render_boss_text(boss_payload))
            return
        if as_json:
            print(json.dumps(run_payload, indent=2))
        else:
            _print_supervisor_run(run_payload)

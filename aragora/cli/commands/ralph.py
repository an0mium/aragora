"""CLI command handler for ``aragora ralph campaign-supervisor``."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cmd_ralph(args) -> None:
    """Dispatch ralph subcommands."""
    action = getattr(args, "ralph_action", "campaign-supervisor")
    subaction = getattr(args, "ralph_subaction", "status")

    if action != "campaign-supervisor":
        print(f"Unknown ralph action: {action}", file=sys.stderr)
        print("Usage: aragora ralph campaign-supervisor [start|step|status|stop|resume]")
        sys.exit(1)

    json_output = getattr(args, "json_output", False)
    state_path = Path(getattr(args, "state", ".aragora/supervisor_state.yaml"))
    repo_root = Path.cwd()

    if subaction == "start":
        _cmd_start(args, state_path, repo_root, json_output)
    elif subaction == "step":
        _cmd_step(state_path, repo_root, args, json_output)
    elif subaction == "status":
        _cmd_status(state_path, json_output)
    elif subaction == "stop":
        _cmd_stop(state_path, repo_root, json_output)
    elif subaction == "resume":
        _cmd_step(state_path, repo_root, args, json_output)
    else:
        print(f"Unknown subaction: {subaction}", file=sys.stderr)
        print("Usage: aragora ralph campaign-supervisor [start|step|status|stop|resume]")
        sys.exit(1)


def _cmd_start(args, state_path: Path, repo_root: Path, json_output: bool) -> None:
    from aragora.ralph.supervisor import RalphSupervisor

    manifest_path_str = getattr(args, "manifest", None)
    if not manifest_path_str:
        print("Error: --manifest is required for start", file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(manifest_path_str)
    merge_policy = getattr(args, "merge_policy", "manual_review_required")
    max_repair = getattr(args, "max_repair_attempts", 2)

    supervisor = RalphSupervisor.start(
        manifest_path=manifest_path,
        state_path=state_path,
        repo_root=repo_root,
        merge_policy=merge_policy,
        max_repair_attempts=max_repair,
    )
    state = supervisor.status()

    if json_output:
        print(json.dumps(state, indent=2))
    else:
        print(f"Ralph supervisor started: {state['supervisor_id']}")
        print(f"  Campaign: {state['campaign_id']}")
        print(f"  State file: {state_path}")
        print(f"  Status: {state['status']}")
        print()
        print("Run next step: aragora ralph campaign-supervisor step")


def _cmd_step(state_path: Path, repo_root: Path, args, json_output: bool) -> None:
    from aragora.ralph.supervisor import RalphSupervisor

    merge_policy = getattr(args, "merge_policy", "manual_review_required")

    supervisor = RalphSupervisor(
        state_path=state_path,
        repo_root=repo_root,
        merge_policy=merge_policy,
    )
    result = supervisor.step()

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Step result: {result.action}")
        print(f"  Status: {result.status}")
        if result.detail:
            print(f"  Detail: {result.detail}")
        if result.repair_task:
            print(f"  Repair task: {result.repair_task.title}")
            print(f"  Allowed paths: {', '.join(result.repair_task.allowed_paths)}")
            print(f"  Done condition: {result.repair_task.done_condition}")


def _cmd_status(state_path: Path, json_output: bool) -> None:
    from aragora.ralph.supervisor import load_supervisor_state

    try:
        state = load_supervisor_state(state_path)
    except FileNotFoundError:
        if json_output:
            print(json.dumps({"error": "No supervisor state found", "path": str(state_path)}))
        else:
            print(f"No supervisor state found at {state_path}")
            print("Start one with: aragora ralph campaign-supervisor start --manifest <path>")
        sys.exit(1)

    d = state.to_dict()
    if json_output:
        print(json.dumps(d, indent=2))
    else:
        print(f"Supervisor: {d['supervisor_id']}")
        print(f"  Campaign: {d['campaign_id']}")
        print(f"  Status: {d['status']}")
        print(f"  Step: {d['current_step']}")
        print(f"  Budget spent: ${d['budget_spent_usd']:.2f}")
        if d.get("active_blocker"):
            print(f"  Active blocker: {d['active_blocker']}")
        task = d.get("active_repair_task") if isinstance(d.get("active_repair_task"), dict) else {}
        if task.get("run_id"):
            print(f"  Repair run: {task['run_id']}")
        if d.get("active_repair_branch"):
            print(f"  Repair branch: {d['active_repair_branch']}")
        if d.get("active_repair_pr"):
            print(f"  Repair PR: {d['active_repair_pr']}")
        if d.get("escalation_reason"):
            print(f"  Escalation: {d['escalation_reason']}")
        print(f"  Last stop reason: {d.get('last_stop_reason', 'none')}")
        print(f"  Repair attempts: {d['repair_attempts']}/{d['max_repair_attempts']}")
        print(f"  Updated: {d['updated_at']}")


def _cmd_stop(state_path: Path, repo_root: Path, json_output: bool) -> None:
    from aragora.ralph.supervisor import RalphSupervisor

    supervisor = RalphSupervisor(state_path=state_path, repo_root=repo_root)
    result = supervisor.stop()

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Supervisor stopped: {result.detail}")

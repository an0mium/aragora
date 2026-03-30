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
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from uuid import uuid4


def _resolve_swarm_action_goal(args: argparse.Namespace) -> tuple[str, str | None]:
    first = getattr(args, "swarm_action_or_goal", None)
    second = getattr(args, "swarm_goal", None)
    if first in {
        "run",
        "boss",
        "boss-loop",
        "audit-issues",
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


def _parse_csv_set(value: object) -> set[str] | None:
    text = _optional_text(value)
    if not text:
        return None
    result = {item.strip() for item in text.split(",") if item.strip()}
    return result or None


def _format_elapsed_seconds(value: object) -> str:
    if value is None:
        return "-"
    seconds = max(0.0, float(value))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        remainder = int(seconds % 60)
        return f"{minutes}m {remainder}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _print_table(
    headers: list[tuple[str, str]],
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        print("(no items)")
        return
    widths = {
        key: max(len(label), *(len(str(row.get(key, "") or "")) for row in rows))
        for key, label in headers
    }
    print("  ".join(label.ljust(widths[key]) for key, label in headers))
    print("  ".join("-" * widths[key] for key, _label in headers))
    for row in rows:
        print("  ".join(str(row.get(key, "") or "").ljust(widths[key]) for key, _label in headers))


def _trim_command_output(text: str, *, limit: int = 240) -> str | None:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return None
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


_UNSAFE_VALIDATION_SHELL_FRAGMENTS = (
    "|",
    "&&",
    "||",
    ";",
    "<",
    ">",
    "$(",
    "`",
    "\n",
    "\r",
)


def _probe_validation_command(
    command: str,
    *,
    repo_root: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    normalized = str(command or "").strip()
    if not normalized:
        return {
            "command": command,
            "status": "unsafe",
            "detail": "empty validation command",
        }
    for fragment in _UNSAFE_VALIDATION_SHELL_FRAGMENTS:
        if fragment in normalized:
            return {
                "command": command,
                "status": "unsafe",
                "detail": (
                    "shell operators are not allowed in auto-probed validation commands; "
                    "use a single direct command instead"
                ),
            }
    try:
        argv = shlex.split(normalized, posix=True)
    except ValueError as exc:
        return {
            "command": command,
            "status": "unsafe",
            "detail": f"invalid shell quoting: {exc}",
        }
    if not argv:
        return {
            "command": command,
            "status": "unsafe",
            "detail": "empty validation command",
        }
    try:
        proc = subprocess.run(
            argv,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "status": "timeout",
            "stdout": _trim_command_output(getattr(exc, "stdout", "") or ""),
            "stderr": _trim_command_output(getattr(exc, "stderr", "") or ""),
        }
    except (FileNotFoundError, OSError) as exc:
        return {
            "command": command,
            "status": "error",
            "stderr": _trim_command_output(str(exc)),
        }

    return {
        "command": command,
        "status": "passed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout": _trim_command_output(proc.stdout),
        "stderr": _trim_command_output(proc.stderr),
    }


def _classify_issue_validation_status(
    *,
    validation_contract: list[str],
    commands: list[str],
    probe_results: list[dict[str, object]],
) -> tuple[str, str]:
    if not validation_contract:
        return (
            "missing_validation_contract",
            "Add an Acceptance Criteria, Validation, or Test section with at least one concrete command.",
        )
    if not commands:
        return (
            "non_runnable_validation_contract",
            "Rewrite the validation contract so it contains runnable commands instead of prose.",
        )
    if probe_results and all(str(item.get("status")) == "passed" for item in probe_results):
        return (
            "passes_now",
            "Issue validations already pass on the current branch; close, relabel, or rewrite the stale queue item.",
        )
    first_failure = next(
        (item for item in probe_results if str(item.get("status")) != "passed"),
        None,
    )
    command = str(first_failure.get("command") if isinstance(first_failure, dict) else "")
    returncode = (
        int(first_failure.get("returncode"))
        if isinstance(first_failure, dict) and isinstance(first_failure.get("returncode"), int)
        else None
    )
    stdout_text = str(first_failure.get("stdout") if isinstance(first_failure, dict) else "" or "")
    stderr_text = str(first_failure.get("stderr") if isinstance(first_failure, dict) else "" or "")
    combined_output = f"{stdout_text}\n{stderr_text}".lower()
    if isinstance(first_failure, dict) and str(first_failure.get("status")) == "unsafe":
        return (
            "unsafe_validation_contract",
            "Rewrite the validation contract as a single direct command without shell pipelines, redirects, or chaining.",
        )
    if command.startswith("python3 -m aragora.cli.main ") and (
        returncode in {1, 2}
        and ("unrecognized arguments" in combined_output or "usage: main.py" in combined_output)
    ):
        return (
            "cli_usage_failure",
            "The current Aragora parser rejects this queued CLI command. Refresh the contract if the flags were renamed, or keep the issue queued if it is meant to add that CLI surface.",
        )
    if returncode == 5 and (
        command.startswith("pytest ")
        or command.startswith("python -m pytest ")
        or command.startswith("python3 -m pytest ")
    ):
        return (
            "no_matching_tests_collected",
            "The queued pytest selector collects no tests on the current branch. Refresh the selector if tests moved, or keep the issue queued if the expected coverage has not been added yet.",
        )
    if any(
        cmd.startswith(prefix)
        for cmd in commands
        for prefix in (
            "pytest tests/ -q",
            "python -m pytest tests/ -q",
            "python3 -m pytest tests/ -q",
        )
    ):
        return (
            "broad_validation_contract",
            "Replace the broad test-suite command with focused validation tied to the intended file scope.",
        )
    return (
        "validation_fails_now",
        "Validation still fails on the current branch; confirm whether the issue remains real or the contract is stale.",
    )


def _audit_issue_validation_contract(
    issue: object,
    *,
    repo_root: Path,
    timeout_seconds: float = 45.0,
) -> dict[str, object]:
    from aragora.swarm.boss_loop import (
        extract_issue_validation_contract,
        extract_pre_dispatch_validation_commands,
    )

    body = str(getattr(issue, "body", "") or "")
    validation_contract = extract_issue_validation_contract(body)
    commands = extract_pre_dispatch_validation_commands(body)
    probe_results: list[dict[str, object]] = []

    for command in commands:
        result = _probe_validation_command(
            command,
            repo_root=repo_root,
            timeout_seconds=timeout_seconds,
        )
        probe_results.append(result)
        if result["status"] != "passed":
            break

    status, next_action = _classify_issue_validation_status(
        validation_contract=validation_contract,
        commands=commands,
        probe_results=probe_results,
    )
    return {
        "number": int(getattr(issue, "number", 0) or 0),
        "title": str(getattr(issue, "title", "") or "").strip(),
        "url": str(getattr(issue, "url", "") or "").strip(),
        "labels": list(getattr(issue, "labels", []) or []),
        "validation_contract": validation_contract,
        "commands": commands,
        "probe_results": probe_results,
        "status": status,
        "next_action": next_action,
    }


@contextmanager
def _open_audit_checkout(repo_root: Path, *, git_ref: str | None):
    if not git_ref:
        yield repo_root
        return

    with tempfile.TemporaryDirectory(prefix="aragora-audit-") as temp_dir:
        checkout_root = Path(temp_dir) / "checkout"
        add_proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "--detach",
                str(checkout_root),
                git_ref,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if add_proc.returncode != 0:
            stderr = _trim_command_output(add_proc.stderr or "") or "git worktree add failed"
            raise RuntimeError(f"Failed to open audit checkout for {git_ref}: {stderr}")
        try:
            yield checkout_root
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "worktree",
                    "remove",
                    "--force",
                    str(checkout_root),
                ],
                text=True,
                capture_output=True,
                check=False,
            )


def _build_runner_report_payload(
    *,
    registrations: list[dict[str, object]],
    routing: dict[str, object],
    discovered: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    by_type: dict[str, dict[str, int]] = {}
    by_cost: dict[str, int] = {}
    fresh_count = 0
    probe_failed = 0
    execution_verified = 0
    for item in registrations:
        runner_type = str(item.get("runner_type", "") or "").strip() or "unknown"
        cost_class = str(item.get("cost_class", "") or "").strip() or "local"
        freshness = str(item.get("freshness_status", "") or "").strip() or "unknown"
        probe_status = str(item.get("probe_status", "") or "").strip() or None
        capabilities = dict(item.get("capabilities") or {})
        max_parallel = int(capabilities.get("max_parallel_lanes") or 1)
        claimed_lanes = int(item.get("claimed_lanes") or 0)
        active_lanes = int(capabilities.get("active_lanes") or item.get("active_lanes") or 0)
        active_lanes += claimed_lanes
        available_capacity = max(0, max_parallel - active_lanes)
        if freshness == "fresh":
            fresh_count += 1
        if probe_status == "passed":
            execution_verified += 1
        elif probe_status == "failed":
            probe_failed += 1
        by_type.setdefault(
            runner_type,
            {
                "registered": 0,
                "fresh": 0,
                "execution_verified": 0,
                "probe_failed": 0,
                "active_lanes": 0,
                "available_capacity": 0,
            },
        )
        by_type[runner_type]["registered"] += 1
        if freshness == "fresh":
            by_type[runner_type]["fresh"] += 1
        if probe_status == "passed":
            by_type[runner_type]["execution_verified"] += 1
        elif probe_status == "failed":
            by_type[runner_type]["probe_failed"] += 1
        by_type[runner_type]["active_lanes"] += active_lanes
        by_type[runner_type]["available_capacity"] += available_capacity
        by_cost[cost_class] = by_cost.get(cost_class, 0) + 1
        rows.append(
            {
                "runner_id": str(item.get("runner_id", "") or "").strip(),
                "runner_type": runner_type,
                "freshness_status": freshness,
                "cost_class": cost_class,
                "probe_status": probe_status,
                "active_lanes": active_lanes,
                "available_capacity": available_capacity,
            }
        )
    rows.sort(key=lambda row: (str(row["runner_type"]), str(row["runner_id"])))
    type_rows = [
        {"runner_type": key, **value}
        for key, value in sorted(by_type.items(), key=lambda item: item[0])
    ]
    cost_rows = [
        {"cost_class": key, "registered": value}
        for key, value in sorted(by_cost.items(), key=lambda item: item[0])
    ]
    discovered_rows = [dict(item) for item in discovered or [] if isinstance(item, dict)]
    selected_runners = [
        item for item in routing.get("selected_runners", []) if isinstance(item, dict)
    ]
    return {
        "mode": "runner",
        "action": "report",
        "summary": {
            "registered": len(registrations),
            "fresh": fresh_count,
            "execution_verified": execution_verified,
            "probe_failed": probe_failed,
            "discovered": len(discovered_rows),
            "selected_for_routing": len(selected_runners),
            "selected_verified": len(
                [
                    item
                    for item in selected_runners
                    if str(item.get("probe_status", "")).strip() == "passed"
                ]
            ),
        },
        "by_runner_type": type_rows,
        "by_cost_class": cost_rows,
        "runners": rows,
        "discovered_runners": discovered_rows,
        "routing": routing,
    }


def _render_runner_report(payload: dict[str, object]) -> None:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    print(
        "Runner Report registered={registered} fresh={fresh} discovered={discovered} selected={selected}".format(
            registered=summary.get("registered", 0),
            fresh=summary.get("fresh", 0),
            discovered=summary.get("discovered", 0),
            selected=summary.get("selected_for_routing", 0),
        )
    )
    print()
    print("By runner type")
    _print_table(
        [
            ("runner_type", "runner_type"),
            ("registered", "registered"),
            ("fresh", "fresh"),
            ("active_lanes", "active"),
            ("available_capacity", "available"),
        ],
        [item for item in payload.get("by_runner_type", []) if isinstance(item, dict)],
    )
    print()
    print("Runners")
    _print_table(
        [
            ("runner_id", "runner_id"),
            ("runner_type", "runner_type"),
            ("freshness_status", "freshness"),
            ("cost_class", "cost"),
            ("active_lanes", "active"),
            ("available_capacity", "available"),
        ],
        [item for item in payload.get("runners", []) if isinstance(item, dict)],
    )
    discovered = [item for item in payload.get("discovered_runners", []) if isinstance(item, dict)]
    if discovered:
        print()
        print("Discovered")
        _print_table(
            [
                ("runner_id", "runner_id"),
                ("runner_type", "runner_type"),
                ("profile", "profile"),
                ("auth_mode", "auth_mode"),
                ("availability", "availability"),
                ("status_summary", "status"),
            ],
            discovered,
        )
    routing = payload.get("routing", {}) if isinstance(payload.get("routing"), dict) else {}
    selected = [
        str(item.get("runner_id", "")).strip()
        for item in routing.get("selected_runners", [])
        if isinstance(item, dict) and str(item.get("runner_id", "")).strip()
    ]
    if selected:
        print()
        print(f"Routing preview: {', '.join(selected)}")
    next_action = str(routing.get("next_action", "") or "").strip()
    if next_action:
        print(f"Next: {next_action}")


def _build_multi_runner_payload(
    *,
    subaction: str,
    runners: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "mode": "runner",
        "action": subaction,
        "summary": {
            "count": len(runners),
            "available": len(
                [
                    item
                    for item in runners
                    if str(item.get("availability", "")).strip() == "available"
                    and bool(item.get("available", True))
                ]
            ),
            "registered": len([item for item in runners if bool(item.get("registered"))]),
        },
        "runners": runners,
    }


def _build_runner_probe_payload(
    *,
    subaction: str,
    runners: list[dict[str, object]],
    discovered: list[dict[str, object]],
    routing_before: dict[str, object] | None = None,
    routing_after: dict[str, object] | None = None,
) -> dict[str, object]:
    attempted = len(runners)
    passed = len(
        [item for item in runners if str(item.get("probe_status", "")).strip() == "passed"]
    )
    failed = len(
        [item for item in runners if str(item.get("probe_status", "")).strip() == "failed"]
    )
    payload: dict[str, object] = {
        "mode": "runner",
        "action": subaction,
        "summary": {
            "discovered": len(discovered),
            "attempted": attempted,
            "passed": passed,
            "failed": failed,
        },
        "runners": runners,
        "discovered_runners": discovered,
    }
    if routing_before is not None:
        payload["routing_before"] = routing_before
        payload["summary"]["selected_before"] = len(
            [item for item in routing_before.get("selected_runners", []) if isinstance(item, dict)]
        )
    if routing_after is not None:
        payload["routing_after"] = routing_after
        payload["summary"]["selected_after"] = len(
            [item for item in routing_after.get("selected_runners", []) if isinstance(item, dict)]
        )
    return payload


def _render_tranche_queue_status(payload: dict[str, object]) -> None:
    print(
        "queue_id={queue_id} status={status} current_item_id={current_item_id}".format(
            queue_id=payload.get("queue_id", ""),
            status=payload.get("status", ""),
            current_item_id=payload.get("current_item_id", "") or "none",
        )
    )
    rows: list[dict[str, object]] = []
    for item in [entry for entry in payload.get("items", []) if isinstance(entry, dict)]:
        worker_branches = [
            str(branch).strip() for branch in item.get("worker_branches", []) if str(branch).strip()
        ]
        rows.append(
            {
                "item_id": str(item.get("item_id", "")).strip(),
                "status": str(item.get("status", "")).strip(),
                "pr_url": str(item.get("pr_url", "")).strip() or "-",
                "worker_branch": (
                    str(item.get("worker_branch", "")).strip() or ", ".join(worker_branches) or "-"
                ),
                "elapsed": _format_elapsed_seconds(item.get("elapsed_seconds")),
            }
        )
    print()
    _print_table(
        [
            ("item_id", "item_id"),
            ("status", "status"),
            ("pr_url", "pr_url"),
            ("worker_branch", "worker_branch"),
            ("elapsed", "elapsed"),
        ],
        rows,
    )


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


def _render_tranche_queue_harvest_table(payload: dict[str, object]) -> None:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    rows = [
        ("total items", int(summary.get("total_items", 0) or 0)),
        ("PRs created", int(summary.get("prs_created", 0) or 0)),
        ("completed", int(summary.get("completed", 0) or 0)),
        ("needs_human", int(summary.get("needs_human", 0) or 0)),
        ("failed", int(summary.get("failed", 0) or 0)),
    ]
    metric_width = max(len("metric"), *(len(label) for label, _ in rows))
    count_width = max(len("count"), *(len(str(value)) for _, value in rows))
    queue_id = str(payload.get("queue_id", "") or "").strip()
    status = str(payload.get("status", "") or "").strip()

    title = f"Tranche Queue Harvest ({queue_id})" if queue_id else "Tranche Queue Harvest"
    print(title)
    if status:
        print(f"status={status}")
    print()
    print(f"{'metric':<{metric_width}}  {'count':>{count_width}}")
    print(f"{'-' * metric_width}  {'-' * count_width}")
    for label, value in rows:
        print(f"{label:<{metric_width}}  {value:>{count_width}}")


def _run_supervised_or_report(awaitable: object) -> object | None:
    try:
        return asyncio.run(awaitable)
    except ValueError as exc:
        print(f"Error: {exc}")
        return None


def _probe_limit_arg(args: argparse.Namespace, *, default: int = 1) -> int:
    raw = getattr(args, "probe_limit", default)
    try:
        return max(1, int(raw or default))
    except (TypeError, ValueError):
        return default


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


def _resolve_boss_routing(
    *,
    requested_runner_type: str | None = None,
    allowed_profiles: set[str] | None = None,
    rotation_interval_seconds: float = 1800.0,
) -> dict[str, object]:
    from aragora.swarm.runner_registry import (
        LocalRunnerRegistry,
        authorization_context_with_defaults,
    )

    owner_context = authorization_context_with_defaults(repo_root=Path.cwd())
    return (
        LocalRunnerRegistry()
        .resolve_boss_routing(
            owner_context=owner_context,
            requested_runner_type=requested_runner_type,
            allowed_profiles=allowed_profiles,
            rotation_interval_seconds=rotation_interval_seconds,
        )
        .to_dict()
    )


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
    claude_runner_profiles = _optional_text(getattr(args, "claude_runner_profiles", None))
    if claude_runner_profiles:
        os.environ["ARAGORA_CLAUDE_RUNNER_PROFILES"] = claude_runner_profiles
    allowed_runner_profiles = _parse_csv_set(claude_runner_profiles)
    runner_rotation_interval = float(getattr(args, "runner_rotation_interval", 1800.0) or 1800.0)
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
            LocalRunnerRegistry,
            authorization_context_with_defaults,
            discover_runner_inspections,
            prioritized_probe_candidates,
            probe_runner_execution,
            refresh_discovered_runners,
        )

        subaction = str(goal or "inspect").strip().lower()
        if subaction not in {"inspect", "register", "heartbeat", "report", "probe", "maintain"}:
            print(
                "Error: swarm runner action must be 'inspect', 'register', 'heartbeat', "
                "'report', 'probe', or 'maintain'"
            )
            return

        runner_type = (
            str(
                getattr(args, "runner_type", None) or os.environ.get("ARAGORA_RUNNER_TYPE", "codex")
            ).strip()
            or "codex"
        )
        inspections: list[object] = []
        probe_limit = _probe_limit_arg(args, default=1 if subaction == "maintain" else 2)
        if subaction == "register":
            inspections = discover_runner_inspections(
                runner_type,
                env=os.environ,
                repo_root=Path.cwd(),
            )
            owner_context = authorization_context_with_defaults(repo_root=Path.cwd())
            payloads = [
                LocalRunnerRegistry()
                .register(
                    inspection,
                    owner_context=owner_context,
                )
                .to_dict()
                for inspection in inspections
            ]
            payload = (
                payloads[0]
                if len(payloads) == 1
                else _build_multi_runner_payload(subaction=subaction, runners=payloads)
            )
        elif subaction == "heartbeat":
            inspections = discover_runner_inspections(
                runner_type,
                env=os.environ,
                repo_root=Path.cwd(),
            )
            owner_context = authorization_context_with_defaults(repo_root=Path.cwd())
            payloads = [
                LocalRunnerRegistry()
                .heartbeat(
                    inspection,
                    owner_context=owner_context,
                )
                .to_dict()
                for inspection in inspections
            ]
            payload = (
                payloads[0]
                if len(payloads) == 1
                else _build_multi_runner_payload(subaction=subaction, runners=payloads)
            )
        elif subaction == "report":
            owner_context = authorization_context_with_defaults(repo_root=Path.cwd())
            registry = LocalRunnerRegistry()
            inspections = (
                refresh_discovered_runners(
                    runner_type,
                    registry=registry,
                    owner_context=owner_context,
                    env=os.environ,
                    repo_root=Path.cwd(),
                )
                if owner_context is not None
                else discover_runner_inspections(
                    runner_type,
                    env=os.environ,
                    repo_root=Path.cwd(),
                )
            )
            payload = _build_runner_report_payload(
                registrations=registry.list_registrations(),
                routing=registry.resolve_boss_routing(
                    owner_context=owner_context,
                    requested_runner_type=runner_type,
                    allowed_profiles=allowed_runner_profiles,
                    rotation_interval_seconds=runner_rotation_interval,
                ).to_dict(),
                discovered=[item.to_dict() for item in inspections],
            )
        elif subaction == "probe":
            owner_context = authorization_context_with_defaults(repo_root=Path.cwd())
            registry = LocalRunnerRegistry()
            inspections = discover_runner_inspections(
                runner_type,
                env=os.environ,
                repo_root=Path.cwd(),
            )
            routing_before = (
                registry.resolve_boss_routing(
                    owner_context=owner_context,
                    requested_runner_type=runner_type,
                    allowed_profiles=allowed_runner_profiles,
                    rotation_interval_seconds=runner_rotation_interval,
                ).to_dict()
                if owner_context is not None
                else None
            )
            probe_payloads: list[dict[str, object]] = []
            for inspection in inspections[:probe_limit]:
                probe = probe_runner_execution(
                    inspection,
                    repo_root=Path.cwd(),
                )
                probe_payloads.append(
                    registry.record_probe(
                        inspection,
                        probe,
                        owner_context=owner_context,
                    )
                )
            routing_after = (
                registry.resolve_boss_routing(
                    owner_context=owner_context,
                    requested_runner_type=runner_type,
                    allowed_profiles=allowed_runner_profiles,
                    rotation_interval_seconds=runner_rotation_interval,
                ).to_dict()
                if owner_context is not None
                else None
            )
            payload = _build_runner_probe_payload(
                subaction=subaction,
                runners=probe_payloads,
                discovered=[item.to_dict() for item in inspections],
                routing_before=routing_before,
                routing_after=routing_after,
            )
        elif subaction == "maintain":
            owner_context = authorization_context_with_defaults(repo_root=Path.cwd())
            registry = LocalRunnerRegistry()
            inspections = (
                refresh_discovered_runners(
                    runner_type,
                    registry=registry,
                    owner_context=owner_context,
                    env=os.environ,
                    repo_root=Path.cwd(),
                )
                if owner_context is not None
                else discover_runner_inspections(
                    runner_type,
                    env=os.environ,
                    repo_root=Path.cwd(),
                )
            )
            routing_before = (
                registry.resolve_boss_routing(
                    owner_context=owner_context,
                    requested_runner_type=runner_type,
                    allowed_profiles=allowed_runner_profiles,
                    rotation_interval_seconds=runner_rotation_interval,
                ).to_dict()
                if owner_context is not None
                else None
            )
            candidates = (
                prioritized_probe_candidates(
                    registry=registry,
                    runner_type=runner_type,
                    discovered_inspections=inspections,
                    owner_context=owner_context,
                    selected_runners=(
                        list(routing_before.get("selected_runners", []))
                        if isinstance(routing_before, dict)
                        else None
                    ),
                )
                if owner_context is not None
                else list(inspections)
            )
            if not candidates:
                candidates = list(inspections)
            probe_payloads = []
            for inspection in candidates[:probe_limit]:
                probe = probe_runner_execution(
                    inspection,
                    repo_root=Path.cwd(),
                )
                probe_payloads.append(
                    registry.record_probe(
                        inspection,
                        probe,
                        owner_context=owner_context,
                    )
                )
            routing_after = (
                registry.resolve_boss_routing(
                    owner_context=owner_context,
                    requested_runner_type=runner_type,
                    allowed_profiles=allowed_runner_profiles,
                    rotation_interval_seconds=runner_rotation_interval,
                ).to_dict()
                if owner_context is not None
                else None
            )
            payload = _build_runner_probe_payload(
                subaction=subaction,
                runners=probe_payloads,
                discovered=[item.to_dict() for item in inspections],
                routing_before=routing_before,
                routing_after=routing_after,
            )
        else:
            inspections = discover_runner_inspections(
                runner_type,
                env=os.environ,
                repo_root=Path.cwd(),
            )
            inspection_payloads = [item.to_dict() for item in inspections]
            payload = (
                inspection_payloads[0]
                if len(inspection_payloads) == 1
                else _build_multi_runner_payload(subaction=subaction, runners=inspection_payloads)
            )

        payload["mode"] = "runner"
        payload["action"] = subaction
        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            if subaction == "report":
                _render_runner_report(payload)
            elif subaction in {"probe", "maintain"}:
                summary = (
                    payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
                )
                print(
                    "Runner {action} discovered={discovered} attempted={attempted} "
                    "passed={passed} failed={failed}".format(
                        action=subaction,
                        discovered=summary.get("discovered", 0),
                        attempted=summary.get("attempted", 0),
                        passed=summary.get("passed", 0),
                        failed=summary.get("failed", 0),
                    )
                )
                print()
                for item in [
                    entry for entry in payload.get("runners", []) if isinstance(entry, dict)
                ]:
                    print(render_runner_registration_text(item))
                    print()
            elif "runners" in payload:
                summary = (
                    payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
                )
                print(
                    "Runner {action} count={count} available={available} registered={registered}".format(
                        action=subaction,
                        count=summary.get("count", 0),
                        available=summary.get("available", 0),
                        registered=summary.get("registered", 0),
                    )
                )
                print()
                for item in [
                    entry for entry in payload.get("runners", []) if isinstance(entry, dict)
                ]:
                    print(render_runner_registration_text(item))
                    print()
            else:
                print(render_runner_registration_text(payload))
        return

    if action == "audit-issues":
        from aragora.swarm.boss_loop import GitHubIssueFeed

        cli_labels: list[str] = list(getattr(args, "labels", None) or [])
        audit_ref = _optional_text(getattr(args, "audit_ref", None))
        legacy_label = getattr(args, "boss_label_filter", None)
        if legacy_label and legacy_label not in cli_labels:
            cli_labels.insert(0, legacy_label)
        label_filter = cli_labels[0] if cli_labels else None
        required_labels = set(cli_labels)
        issue_list = [
            int(item.strip())
            for item in str(getattr(args, "boss_issue_list", "") or "").split(",")
            if item.strip()
        ]
        if getattr(args, "boss_issue_number", None):
            issue_list.append(int(getattr(args, "boss_issue_number")))

        feed = GitHubIssueFeed(
            repo=getattr(args, "boss_repo", None),
            label_filter=label_filter,
            issue_numbers=issue_list or None,
            limit=25,
        )
        issues = feed.fetch()
        if required_labels:
            issues = [
                issue
                for issue in issues
                if required_labels.issubset(
                    {str(label).strip() for label in getattr(issue, "labels", [])}
                )
            ]

        with _open_audit_checkout(Path.cwd(), git_ref=audit_ref) as audit_root:
            audits = [
                _audit_issue_validation_contract(issue, repo_root=audit_root) for issue in issues
            ]
        summary: dict[str, int] = {}
        for item in audits:
            status = str(item.get("status", "unknown") or "unknown")
            summary[status] = summary.get(status, 0) + 1
        payload = {
            "mode": "swarm-issue-audit",
            "action": "audit-issues",
            "repo": getattr(args, "boss_repo", None),
            "audit_ref": audit_ref,
            "labels": cli_labels,
            "issue_count": len(audits),
            "summary": summary,
            "issues": audits,
        }
        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"audited={len(audits)} "
                + (f"ref={audit_ref} " if audit_ref else "")
                + " ".join(f"{key}={value}" for key, value in sorted(summary.items()))
            )
            rows = [
                {
                    "number": item["number"],
                    "status": item["status"],
                    "title": item["title"][:64],
                    "next_action": str(item["next_action"])[:80],
                }
                for item in audits
            ]
            _print_table(
                [
                    ("number", "Issue"),
                    ("status", "Status"),
                    ("title", "Title"),
                    ("next_action", "Next Action"),
                ],
                rows,
            )
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

        # Merge --label (repeatable) with legacy --boss-label-filter (single string).
        cli_labels: list[str] = list(getattr(args, "labels", None) or [])
        legacy_label = getattr(args, "boss_label_filter", None)
        if legacy_label and legacy_label not in cli_labels:
            cli_labels.insert(0, legacy_label)

        # Use the first label for gh CLI pre-filtering (server-side), and the
        # full set as require_labels for Python-side ALL-match filtering.
        label_filter = cli_labels[0] if cli_labels else None
        require_labels = set(cli_labels) if cli_labels else None

        # When --autonomy full-auto, continue past needs_human states
        auto_continue = autonomy_str in {"full-auto", "fire_and_forget"}
        issue_list = [
            int(item.strip())
            for item in str(getattr(args, "boss_issue_list", "") or "").split(",")
            if item.strip()
        ]
        default_target_agent = str(getattr(args, "worker_model", "") or "").strip() or None
        default_reviewer_agent = str(getattr(args, "review_model", "") or "").strip() or None

        boss_loop_config = BossLoopConfig(
            max_iterations=int(getattr(args, "max_ticks", None) or 50),
            iteration_interval_seconds=float(getattr(args, "interval_seconds", 30.0) or 30.0),
            freshness_ttl_seconds=float(getattr(args, "freshness_ttl", 3600.0) or 3600.0),
            repo=getattr(args, "boss_repo", None),
            label_filter=label_filter,
            require_labels=require_labels,
            issue_number=getattr(args, "boss_issue_number", None),
            issue_numbers=issue_list or None,
            target_branch=target_branch,
            budget_limit_usd=budget_limit,
            max_consecutive_failures=int(getattr(args, "max_consecutive_failures", 3) or 3),
            require_validation_contract=not bool(
                getattr(args, "allow_missing_validation_contract", False)
            ),
            dispatch_enabled=not no_dispatch,
            default_target_agent=default_target_agent,
            default_reviewer_agent=default_reviewer_agent,
            allowed_runner_profiles=allowed_runner_profiles,
            runner_rotation_interval_seconds=runner_rotation_interval,
            max_parallel_dispatches=int(getattr(args, "boss_max_parallel_dispatches", 1) or 1),
            auto_continue_on_needs_human=auto_continue,
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
            integrate_lane,
        )
        from aragora.swarm.tranche_queue import (
            compile_tranche_queue,
            harvest_tranche_queue,
            reconcile_tranche_queue,
            run_tranche_queue,
            tranche_queue_status,
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
        if subaction == "compile-queue":
            sources_arg = str(getattr(args, "sources", "") or "").strip()
            if not sources_arg:
                raise ValueError("tranche compile-queue requires --sources <path>")
            output_arg = str(getattr(args, "output", "") or "").strip()
            if not output_arg:
                raise ValueError("tranche compile-queue requires --output <path>")
            sources_path = Path(sources_arg).resolve()
            if not sources_path.exists():
                raise ValueError(f"tranche queue source manifest not found: {sources_path}")
            output_path = Path(output_arg).resolve()
            payload = compile_tranche_queue(
                sources_path=sources_path,
                output_path=output_path,
                repo_root=repo_root,
            )
            payload["action"] = subaction
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(json.dumps(payload, indent=2))
            return
        if subaction == "status":
            queue_arg = str(getattr(args, "queue", "") or "").strip()
            if not queue_arg:
                raise ValueError("tranche status requires --queue <path>")
            queue_path = Path(queue_arg).resolve()
            if not queue_path.exists():
                raise ValueError(f"tranche queue manifest not found: {queue_path}")
            payload = tranche_queue_status(
                queue_path=queue_path,
                repo_root=repo_root,
            )
            payload["action"] = subaction
            payload["queue_path"] = str(queue_path)
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                _render_tranche_queue_status(payload)
            return
        if subaction in {"run-queue", "reconcile-queue", "harvest-queue"}:
            queue_arg = str(getattr(args, "queue", "") or "").strip()
            if not queue_arg:
                raise ValueError(f"tranche {subaction} requires --queue <path>")
            queue_path = Path(queue_arg).resolve()
            if not queue_path.exists():
                raise ValueError(f"tranche queue manifest not found: {queue_path}")
            if subaction == "run-queue":
                payload = asyncio.run(
                    run_tranche_queue(
                        queue_path=queue_path,
                        repo_root=repo_root,
                        target_branch=str(getattr(args, "target_branch", "main") or "main"),
                        interval_seconds=interval_seconds,
                        max_hours=float(getattr(args, "max_hours", 12.0) or 12.0),
                        max_consecutive_failures=int(
                            getattr(args, "max_consecutive_failures", 3) or 3
                        ),
                        planner_model=str(getattr(args, "planner_model", "claude") or "claude"),
                        planner_strategy=str(
                            getattr(args, "planner_strategy", "heuristic") or "heuristic"
                        ),
                        worker_model=str(getattr(args, "worker_model", "codex") or "codex"),
                        review_model=str(getattr(args, "review_model", "claude") or "claude"),
                        max_parallel_lanes=int(getattr(args, "max_parallel_lanes", 1) or 1),
                        enforce_cross_model_review=not bool(
                            getattr(args, "allow_same_model_review", False)
                        ),
                    )
                )
            elif subaction == "harvest-queue":
                payload = harvest_tranche_queue(
                    queue_path=queue_path,
                    repo_root=repo_root,
                    execute_merge=bool(getattr(args, "execute_merge", False)),
                    allow_admin=bool(getattr(args, "allow_admin", False)),
                )
            else:
                payload = reconcile_tranche_queue(
                    queue_path=queue_path,
                    repo_root=repo_root,
                )
            payload["action"] = subaction
            payload["queue_path"] = str(queue_path)
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                if subaction == "harvest-queue":
                    _render_tranche_queue_harvest_table(payload)
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
            executor = TrancheExecutor(repo_root=repo_root) if driver_mode else None
            supervisor = None
            github = None
            registry = None

            async def _watch_run_fn(*, manifest):
                if executor is None:
                    return None
                try:
                    return await executor.run(
                        manifest,
                        owner_session_id=session_id,
                        target_branch=str(getattr(args, "target_branch", "main") or "main"),
                        max_ticks=int(getattr(args, "max_ticks", 360) or 360),
                        wait_for_completion=False,
                        skip_review=True,
                    )
                except ValueError as exc:
                    detail = str(exc or "").strip()
                    if (
                        "No ready claimable lanes found" in detail
                        or "Tranche is not ready to run." in detail
                        or detail.endswith("is not ready.")
                    ):
                        return None
                    raise

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

            async def _watch_integrate_fn(*, manifest, lane_id, artifact, approve, run_state=None):
                nonlocal github, registry
                if artifact is None:
                    return {"recommendation": "needs_human", "executed": False}
                if github is None:
                    github = GitHubControl(repo_root=repo_root)
                if registry is None:
                    registry = PullRequestRegistry()
                coord_store = DevCoordinationStore(repo_root=repo_root)
                return await integrate_lane(
                    artifact=artifact,
                    manifest=manifest,
                    approve=bool(approve),
                    repo_root=repo_root,
                    github=github,
                    registry=registry,
                    store=coord_store,
                    target_branch=str(getattr(args, "target_branch", "main") or "main"),
                    decided_by="tranche-watch",
                    rationale="Tranche watch approved merge after green checks and review.",
                    run_state=run_state,
                    autonomy_mode=str(state.autonomy_mode or "adaptive"),
                )

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
                    driver_session_id=session_id if driver_mode else None,
                    artifact_store=artifact_store,
                    repo_root=repo_root,
                    run_fn=_watch_run_fn if driver_mode else None,
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
            state_path = run_state_path_for_manifest(manifest_path)
            run_state = None
            try:
                if state_path.exists():
                    run_state = load_tranche_run_state(manifest_path)
            except (OSError, ValueError):
                run_state = None
            results: list[dict[str, object]] = []
            for artifact in selected_artifacts:
                result = asyncio.run(
                    integrate_lane(
                        manifest=manifest,
                        artifact=artifact,
                        approve=approve,
                        repo_root=repo_root,
                        github=github,
                        registry=registry,
                        store=store,
                        artifact_store=artifact_store,
                        target_branch=str(getattr(args, "target_branch", "main") or "main"),
                        decided_by=str(getattr(args, "decided_by", None) or "tranche-integrate"),
                        rationale=str(
                            getattr(args, "rationale", None)
                            or "Tranche integrate approved merge after green checks and review."
                        ),
                        run_state=run_state,
                        autonomy_mode=str(getattr(args, "autonomy", "adaptive") or "adaptive"),
                    )
                )
                results.append(result)

            if run_state is not None:
                run_state.save(state_path)

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
                "tranche action must be one of: submit, plan, inspect, watch, list, design-review, review, integrate, prepare, run, status, compile-queue, run-queue, reconcile-queue, harvest-queue"
            )
        payload["action"] = subaction
        payload["manifest_path"] = str(manifest_path)
        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return

    if boss_mode:
        boss_routing = _resolve_boss_routing(
            requested_runner_type=str(getattr(args, "worker_model", "") or "").strip() or None,
            allowed_profiles=allowed_runner_profiles,
            rotation_interval_seconds=runner_rotation_interval,
        )
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

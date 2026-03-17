#!/usr/bin/env python3
"""Support tooling for the Phase 0B Codex/Claude role benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aragora.swarm.campaign import (
    CampaignManifest,
    CampaignProject,
    CampaignProjectStatus,
    CampaignReviewGate,
    CampaignReviewStatus,
    load_campaign_manifest,
    save_campaign_manifest,
)

UTC = timezone.utc
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "docs" / "plans" / "phase0b_campaign_manifest.yaml"
EXPERIMENT_DIR = PROJECT_ROOT / "docs" / "experiments" / "phase0b_role_benchmark"
RUNS_DIR = EXPERIMENT_DIR / "runs"
ACTIVE_RUN_PATH = EXPERIMENT_DIR / "active_run.json"
RESULTS_JSON_PATH = EXPERIMENT_DIR / "results.json"
RESULTS_CSV_PATH = EXPERIMENT_DIR / "results.csv"
DEFAULT_RUNTIME_MANIFEST = Path(".aragora/phase0b_runtime_manifest.yaml")
RESULT_COLUMNS = [
    "recorded_at",
    "experiment_id",
    "experiment_label",
    "config_id",
    "campaign_id",
    "project_id",
    "runtime_path",
    "runtime_manifest_path",
    "planner_model",
    "planner_strategy",
    "worker_model",
    "review_model",
    "enforce_cross_model_review",
    "project_status",
    "last_run_outcome",
    "review_status",
    "worker_branch",
    "worker_commit",
    "worker_branch_count",
    "worker_commit_count",
    "worker_branches_json",
    "worker_commits_json",
    "pr_url",
    "pr_state",
    "ci_status",
    "duration_seconds",
    "cost_usd",
    "changed_files_count",
    "planner_strategy_requested",
    "planner_strategy_used",
    "planner_fallback_reason",
    "verification_missing_reason",
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _config_id(planner_model: str, worker_model: str, review_model: str) -> str:
    return f"p-{planner_model}_w-{worker_model}_r-{review_model}"


def _ensure_layout() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_JSON_PATH.exists():
        RESULTS_JSON_PATH.write_text(json.dumps({"runs": []}, indent=2), encoding="utf-8")
    if not RESULTS_CSV_PATH.exists():
        with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _active_run() -> dict[str, Any] | None:
    payload = _load_json(ACTIVE_RUN_PATH, None)
    return payload if isinstance(payload, dict) and payload else None


def _load_receipt(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text)
    except ImportError:
        payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


def _gh_json(cmd: list[str]) -> Any:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=30)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None


def _gh_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=30)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _prepare_project(project: CampaignProject) -> CampaignProject:
    prepared = CampaignProject.from_dict(project.to_dict())
    prepared.status = CampaignProjectStatus.PENDING.value
    prepared.retry_count = 0
    prepared.last_run_outcome = None
    prepared.run_id = None
    prepared.worker_receipt_id = None
    prepared.receipt_id = None
    prepared.pr_url = None
    prepared.adopted_pr = None
    prepared.branch = None
    prepared.commit_shas = []
    prepared.attempt_history = []
    prepared.dependencies = []
    prepared.review = CampaignReviewGate(
        required=True,
        review_model=prepared.review.review_model,
        status=CampaignReviewStatus.PENDING.value,
        findings=[],
        reviewed_at=None,
        raw_review={},
    )
    return prepared


def prepare_runtime_manifest(
    *,
    source_manifest: Path,
    worktree: Path,
    project_id: str,
    planner_model: str,
    planner_strategy: str,
    worker_model: str,
    review_model: str,
    enforce_cross_model_review: bool,
    experiment_id: str,
    experiment_label: str | None,
    budget_limit_usd: float | None,
    time_limit_hours: float | None,
) -> dict[str, Any]:
    manifest = load_campaign_manifest(source_manifest)
    project_map = manifest.project_map()
    if project_id not in project_map:
        raise KeyError(f"Unknown project_id {project_id!r} in {source_manifest}")

    prepared_project = _prepare_project(project_map[project_id])
    runtime_manifest = CampaignManifest(
        campaign_id=manifest.campaign_id,
        created_at=manifest.created_at,
        source_kind=manifest.source_kind,
        source_ref=manifest.source_ref,
        planner_model=planner_model,
        planner_strategy=planner_strategy,
        worker_model=worker_model,
        review_model=review_model,
        enforce_cross_model_review=enforce_cross_model_review,
        experiment_id=experiment_id,
        experiment_label=experiment_label,
        max_parallel_ready_projects=1,
        max_retries_per_project=manifest.max_retries_per_project,
        budget_limit_usd=budget_limit_usd
        if budget_limit_usd is not None
        else manifest.budget_limit_usd,
        time_limit_hours=time_limit_hours
        if time_limit_hours is not None
        else manifest.time_limit_hours,
        projects=[prepared_project],
        planning_findings=list(manifest.planning_findings),
        manifest_version=manifest.manifest_version,
    )
    runtime_path = worktree / DEFAULT_RUNTIME_MANIFEST
    save_campaign_manifest(runtime_path, runtime_manifest)

    config_id = _config_id(planner_model, worker_model, runtime_manifest.review_model)
    run_record = {
        "recorded_at": _now_iso(),
        "experiment_id": experiment_id,
        "experiment_label": experiment_label,
        "config_id": config_id,
        "project_id": project_id,
        "runtime_path": str(worktree),
        "runtime_manifest_path": str(runtime_path),
    }
    _write_json(RUNS_DIR / experiment_id / f"{config_id}.json", run_record)
    return {
        **run_record,
        "runtime_manifest_path": str(runtime_path),
        "effective_review_model": runtime_manifest.review_model,
    }


def _lookup_pr(branch: str | None) -> dict[str, Any]:
    if not branch:
        return {}
    payload = _gh_json(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--json",
            "number,url,state",
            "--limit",
            "1",
        ]
    )
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first
    return {}


def _lookup_ci_status(pr_number: int | None) -> str:
    if pr_number is None:
        return ""
    text = _gh_text(["gh", "pr", "checks", str(pr_number)])
    if not text:
        return ""
    lowered = text.lower()
    if " fail " in lowered or lowered.startswith("fail"):
        return "failing"
    if " pending " in lowered or "pending" in lowered:
        return "pending"
    if " pass " in lowered or lowered.startswith("pass"):
        return "passing"
    return "mixed"


def build_result_row(runtime_manifest_path: Path) -> dict[str, Any]:
    manifest = load_campaign_manifest(runtime_manifest_path)
    if not manifest.projects:
        raise ValueError(f"No projects in runtime manifest {runtime_manifest_path}")
    project = manifest.projects[0]
    receipt_path = PROJECT_ROOT / project.receipt_id if project.receipt_id else None
    receipt = _load_receipt(receipt_path)
    branch = str(receipt.get("worker_branch") or project.branch or "").strip() or None
    worker_branches = [
        str(item).strip() for item in receipt.get("worker_branches", []) if str(item).strip()
    ]
    if not worker_branches and branch:
        worker_branches = [branch]
    worker_commit = str(receipt.get("worker_commit") or "").strip()
    worker_commits = [
        str(item).strip() for item in receipt.get("worker_commits", []) if str(item).strip()
    ]
    if not worker_commits and worker_commit:
        worker_commits = [worker_commit]
    pr_lookup = _lookup_pr(branch)
    pr_number = int(pr_lookup["number"]) if isinstance(pr_lookup.get("number"), int) else None
    review_model = manifest.review_model
    config_id = _config_id(manifest.planner_model, manifest.worker_model, review_model)

    return {
        "recorded_at": _now_iso(),
        "experiment_id": manifest.experiment_id,
        "experiment_label": manifest.experiment_label,
        "config_id": config_id,
        "campaign_id": manifest.campaign_id,
        "project_id": project.project_id,
        "runtime_path": str(runtime_manifest_path.parent.parent),
        "runtime_manifest_path": str(runtime_manifest_path),
        "planner_model": manifest.planner_model,
        "planner_strategy": manifest.planner_strategy,
        "worker_model": manifest.worker_model,
        "review_model": review_model,
        "enforce_cross_model_review": manifest.enforce_cross_model_review,
        "project_status": project.status,
        "last_run_outcome": project.last_run_outcome,
        "review_status": project.review.status,
        "worker_branch": branch or "",
        "worker_commit": worker_commit,
        "worker_branch_count": len(worker_branches),
        "worker_commit_count": len(worker_commits),
        "worker_branches_json": json.dumps(worker_branches),
        "worker_commits_json": json.dumps(worker_commits),
        "pr_url": str(project.pr_url or pr_lookup.get("url") or "").strip(),
        "pr_state": str(pr_lookup.get("state") or "").strip(),
        "ci_status": _lookup_ci_status(pr_number),
        "duration_seconds": receipt.get("duration_seconds"),
        "cost_usd": receipt.get("cost_usd", project.estimated_cost_usd),
        "changed_files_count": len(receipt.get("changed_files") or []),
        "planner_strategy_requested": receipt.get("planner_strategy_requested"),
        "planner_strategy_used": receipt.get("planner_strategy_used"),
        "planner_fallback_reason": receipt.get("planner_fallback_reason"),
        "verification_missing_reason": receipt.get("verification_missing_reason"),
    }


def _upsert_result(row: dict[str, Any]) -> None:
    _ensure_layout()
    payload = _load_json(RESULTS_JSON_PATH, {"runs": []})
    runs = [dict(item) for item in payload.get("runs", []) if isinstance(item, dict)]
    key = (
        str(row.get("experiment_id", "")),
        str(row.get("config_id", "")),
        str(row.get("project_id", "")),
        str(row.get("runtime_manifest_path", "")),
    )
    filtered = [
        item
        for item in runs
        if (
            str(item.get("experiment_id", "")),
            str(item.get("config_id", "")),
            str(item.get("project_id", "")),
            str(item.get("runtime_manifest_path", "")),
        )
        != key
    ]
    filtered.append(dict(row))
    filtered.sort(
        key=lambda item: (
            str(item.get("experiment_id", "")),
            str(item.get("config_id", "")),
            str(item.get("recorded_at", "")),
        )
    )
    _write_json(RESULTS_JSON_PATH, {"runs": filtered})
    with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for item in filtered:
            writer.writerow({key: item.get(key, "") for key in RESULT_COLUMNS})


def cmd_prepare(args: argparse.Namespace) -> int:
    _ensure_layout()
    active = _active_run()
    if active and not args.force:
        raise SystemExit(
            "Active benchmark run is locked. Record or clear it before preparing another run."
        )
    payload = prepare_runtime_manifest(
        source_manifest=args.source_manifest.resolve(),
        worktree=args.worktree.resolve(),
        project_id=args.project_id,
        planner_model=args.planner_model,
        planner_strategy=args.planner_strategy,
        worker_model=args.worker_model,
        review_model=args.review_model,
        enforce_cross_model_review=not args.allow_same_model_review,
        experiment_id=args.experiment_id,
        experiment_label=args.experiment_label,
        budget_limit_usd=args.budget_limit_usd,
        time_limit_hours=args.time_limit_hours,
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    _ensure_layout()
    active = _active_run()
    runtime_manifest = args.runtime_manifest.resolve()
    if active and str(active.get("runtime_manifest_path", "")) != str(runtime_manifest):
        raise SystemExit(
            "Another benchmark runtime is active. Record it before starting a new one."
        )
    payload = _gh_json(
        [
            sys.executable,
            "-m",
            "aragora.cli.main",
            "ralph",
            "campaign-supervisor",
            "start",
            "--manifest",
            str(runtime_manifest),
            "--json",
        ]
    )
    if not isinstance(payload, dict):
        raise SystemExit("Failed to start benchmark supervisor run.")
    active_payload = {
        "recorded_at": _now_iso(),
        "runtime_manifest_path": str(runtime_manifest),
        "runtime_path": str(runtime_manifest.parent.parent),
        "supervisor_id": payload.get("supervisor_id"),
    }
    _write_json(ACTIVE_RUN_PATH, active_payload)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    _ensure_layout()
    active = _active_run()
    runtime_manifest = (
        args.runtime_manifest.resolve()
        if args.runtime_manifest
        else Path(str(active.get("runtime_manifest_path", ""))).resolve()
        if active
        else None
    )
    if runtime_manifest is None:
        raise SystemExit("No runtime manifest supplied and no active benchmark run is locked.")
    row = build_result_row(runtime_manifest)
    _upsert_result(row)
    if active and str(active.get("runtime_manifest_path", "")) == str(runtime_manifest):
        ACTIVE_RUN_PATH.unlink(missing_ok=True)
    print(json.dumps(row, indent=2))
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    _ensure_layout()
    active = _active_run()
    payload = {
        "active_run": active,
        "results_path": str(RESULTS_JSON_PATH),
        "csv_path": str(RESULTS_CSV_PATH),
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a fresh benchmark runtime manifest")
    prepare.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    prepare.add_argument("--worktree", type=Path, required=True)
    prepare.add_argument("--project-id", required=True)
    prepare.add_argument("--planner-model", choices=("codex", "claude"), required=True)
    prepare.add_argument("--planner-strategy", choices=("heuristic", "model"), default="model")
    prepare.add_argument("--worker-model", choices=("codex", "claude"), required=True)
    prepare.add_argument("--review-model", choices=("codex", "claude"), required=True)
    prepare.add_argument("--allow-same-model-review", action="store_true")
    prepare.add_argument("--experiment-id", required=True)
    prepare.add_argument("--experiment-label", default=None)
    prepare.add_argument("--budget-limit-usd", type=float, default=None)
    prepare.add_argument("--time-limit-hours", type=float, default=None)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    start = subparsers.add_parser("start", help="Start the prepared benchmark runtime")
    start.add_argument("--runtime-manifest", type=Path, required=True)
    start.set_defaults(func=cmd_start)

    record = subparsers.add_parser(
        "record", help="Record a runtime result into the experiment table"
    )
    record.add_argument("--runtime-manifest", type=Path, default=None)
    record.set_defaults(func=cmd_record)

    status = subparsers.add_parser("status", help="Show current benchmark lock/result paths")
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

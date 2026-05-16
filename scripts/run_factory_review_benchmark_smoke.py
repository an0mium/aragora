#!/usr/bin/env python3
"""Build or execute the Factory code-review benchmark smoke run plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "benchmarks" / "factory_review_benchmark_manifest.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "status"
    / "generated"
    / "factory_review_benchmark"
    / "smoke"
    / "planned-run.json"
)
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / ".aragora" / "factory-review-benchmark" / "smoke"
REQUIRED_CASE_FIELDS = {
    "case_id",
    "repo",
    "pr_number",
    "pr_url",
    "title",
    "base_ref",
    "head_ref",
    "head_sha",
    "validation_path",
    "validation_url",
}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a JSON object")
    return raw


def _validate_case(row: object, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"smoke_cases[{index}] must be a JSON object")
    missing = sorted(field for field in REQUIRED_CASE_FIELDS if field not in row)
    if missing:
        raise ValueError(f"smoke_cases[{index}] missing required fields: {', '.join(missing)}")
    for field in REQUIRED_CASE_FIELDS:
        if field == "pr_number":
            if not isinstance(row[field], int) or isinstance(row[field], bool) or row[field] <= 0:
                raise ValueError(f"smoke_cases[{index}].pr_number must be a positive integer")
            continue
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"smoke_cases[{index}].{field} must be a non-empty string")
    return dict(row)


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = manifest.get("smoke_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must include a non-empty smoke_cases list")
    return [_validate_case(row, index) for index, row in enumerate(cases)]


def _artifact_dir(artifact_root: Path, case: dict[str, Any]) -> Path:
    repo_name = str(case["repo"]).split("/")[-1]
    return artifact_root / repo_name / f"pr-{case['pr_number']}"


def build_run_plan(
    manifest: dict[str, Any],
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    reviewer: str | None = None,
) -> dict[str, Any]:
    cases = _validate_manifest(manifest)
    reviewer_args = ["--reviewer", reviewer] if reviewer else []
    run_cases: list[dict[str, Any]] = []

    for case in cases:
        artifact_dir = _artifact_dir(artifact_root, case)
        command = [
            sys.executable,
            "-m",
            "aragora.cli.main",
            "review-pr",
            str(case["pr_url"]),
            "--artifact-dir",
            str(artifact_dir),
            "--no-publish-review",
            "--json",
            *reviewer_args,
        ]
        run_cases.append(
            {
                "case_id": case["case_id"],
                "repo": case["repo"],
                "pr_number": case["pr_number"],
                "pr_url": case["pr_url"],
                "title": case["title"],
                "base_ref": case["base_ref"],
                "head_ref": case["head_ref"],
                "head_sha": case["head_sha"],
                "expected_head_sha": case["head_sha"],
                "validation_path": case["validation_path"],
                "validation_url": case["validation_url"],
                "artifact_dir": str(artifact_dir),
                "command": command,
                "execute_default": False,
                "pre_execute_head_check": {
                    "type": "github_pr_head_sha",
                    "expected_head_sha": case["head_sha"],
                },
            }
        )

    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "benchmark": manifest.get("benchmark", "factory_review_droid_external_smoke"),
        "source": manifest.get("source", {}),
        "guardrails": {
            "mode": "dry_run",
            "no_publish_review": True,
            "external_pr_comments": False,
            "live_routing_change": False,
        },
        "cases": run_cases,
    }


def _current_pr_head_sha(pr_url: str) -> str:
    result = subprocess.run(
        ["gh", "pr", "view", pr_url, "--json", "headRefOid", "--jq", ".headRefOid"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"unable to verify current head for {pr_url}: {result.stderr.strip()}")
    head_sha = result.stdout.strip()
    if not head_sha:
        raise ValueError(f"unable to verify current head for {pr_url}: empty headRefOid")
    return head_sha


def _verify_case_head(case: dict[str, Any]) -> None:
    expected = str(case.get("expected_head_sha") or case.get("head_sha") or "")
    if not expected:
        raise ValueError(f"{case.get('case_id', '<unknown>')} has no expected head SHA")
    actual = _current_pr_head_sha(str(case["pr_url"]))
    if actual != expected:
        raise ValueError(
            f"{case['case_id']} head SHA drifted: expected {expected}, current {actual}"
        )


def execute_run_plan(plan: dict[str, Any]) -> dict[str, Any]:
    executed: list[dict[str, Any]] = []
    for case in plan["cases"]:
        _verify_case_head(case)
        command = list(case["command"])
        started = _utc_now()
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        executed.append(
            {
                "case_id": case["case_id"],
                "started_at": started,
                "finished_at": _utc_now(),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    updated = dict(plan)
    updated["guardrails"] = {**dict(plan.get("guardrails", {})), "mode": "executed"}
    updated["executions"] = executed
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dry-run execution plan for Aragora review-pr against the Factory "
            "code-review benchmark smoke slice."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the generated review-pr commands. Default only writes the plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = _load_manifest(args.manifest)
        plan = build_run_plan(manifest, artifact_root=args.artifact_root, reviewer=args.reviewer)
        if args.execute:
            plan = execute_run_plan(plan)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "cases": len(plan["cases"])}, sort_keys=True))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

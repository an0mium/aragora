#!/usr/bin/env python3
"""Guard Aragora PR review gate workflow policy against drift."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Violation:
    path: str
    message: str


REVIEW_GATE_WORKFLOW = Path(".github/workflows/aragora-review-gate.yml")
MANUAL_REVIEW_WORKFLOW = Path(".github/workflows/aragora-review.yml")


def _load_workflow(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a workflow object")
    return data, text


def _workflow_on(data: dict) -> dict:
    if "on" in data:
        return data["on"]
    if True in data:
        return data[True]
    return {}


def find_review_gate_policy_violations(
    gate_data: dict,
    gate_text: str,
    manual_data: dict,
) -> list[str]:
    violations: list[str] = []

    gate_on = _workflow_on(gate_data)
    pull_request = gate_on.get("pull_request")
    if not isinstance(pull_request, dict):
        violations.append("review gate must trigger on pull_request")
    else:
        if "paths" in pull_request:
            violations.append("review gate pull_request trigger must not define paths filters")
        expected_types = {"opened", "synchronize", "reopened", "ready_for_review"}
        actual_types = set(pull_request.get("types", []))
        missing_types = sorted(expected_types - actual_types)
        for event_type in missing_types:
            violations.append(f"review gate pull_request trigger missing type: {event_type}")

    jobs = gate_data.get("jobs", {})
    if "aragora-review" not in jobs:
        violations.append("review gate must define aragora-review job")
    if "review" not in jobs:
        violations.append("review gate must define review job")
    if "changes" not in jobs:
        violations.append("review gate must define changes job")

    gate_job = jobs.get("aragora-review", {})
    if gate_job.get("if") != "always()":
        violations.append("aragora-review gate job must use if: always()")
    if gate_job.get("needs") != ["changes", "review"]:
        violations.append("aragora-review gate job must need [changes, review]")

    review_job = jobs.get("review", {})
    review_steps = review_job.get("steps", []) if isinstance(review_job, dict) else []
    install_step = next(
        (step for step in review_steps if step.get("name") == "Install Aragora"),
        None,
    )
    review_step = next(
        (step for step in review_steps if step.get("name") == "Run Aragora Review"),
        None,
    )
    if not isinstance(install_step, dict):
        violations.append("review job must define an Install Aragora step")
        install_run = ""
    else:
        install_run = str(install_step.get("run", ""))
    if not isinstance(review_step, dict):
        violations.append("review job must define a Run Aragora Review step")
        review_run = ""
    else:
        review_run = str(review_step.get("run", ""))

    for dependency in (
        '"pydantic>=2,<3"',
        "pydantic-settings",
        "aiohttp",
        "defusedxml",
        "numpy",
        "anthropic",
        "openai",
    ):
        if dependency not in install_run:
            violations.append(
                f"review job install step must install {dependency} for review CLI runtime"
            )
    if "python -m pip install -e ." not in install_run:
        violations.append("review job install step must install the repo in editable mode")
    if 'if [[ ! -f "$review_json" ]]; then' not in review_run:
        violations.append("review gate must handle missing review.json explicitly")
    if review_run.count('echo "review_failed=true" >> "$GITHUB_OUTPUT"') < 2:
        violations.append("review execution must mark command/artifact failures in outputs")
    if "Aragora advisory review command failed" not in review_run:
        violations.append("review execution must surface review command failures as warnings")
    if "Aragora advisory review did not produce" not in review_run:
        violations.append("review execution must surface missing review artifacts as warnings")
    if "python -m aragora.cli.review review" not in review_run:
        violations.append("review execution must invoke the review subcommand")
    if "--output-format json" not in review_run:
        violations.append("review execution must request json output")
    if '--output-dir "$review_output_dir"' not in review_run:
        violations.append("review execution must write review artifacts via --output-dir")
    if "review_exit=$?" not in review_run:
        violations.append("review execution must capture the review command exit code explicitly")
    if 'if [[ "$review_exit" -ne 0 ]]; then' not in review_run:
        violations.append("review execution must handle nonzero review command explicitly")
    if "critical_issues" not in review_run or "high_issues" not in review_run:
        violations.append("review gate must parse the current json artifact schema")
    if "python -m aragora.cli.review" in review_run and "|| true" in review_run:
        violations.append("review execution must not use || true")
    if "Gate on Critical Findings" in gate_text or (
        "::error::Aragora review found ${CRITICAL_COUNT} critical issue(s)" in gate_text
    ):
        violations.append("review findings must stay advisory and must not fail on critical-count")
    if 'if [[ "$REVIEW_RESULT" != "success" ]]' not in gate_text:
        violations.append("gate result must surface unsuccessful review jobs")
    if 'if [[ "$SHOULD_REVIEW" != "true" ]]' not in gate_text:
        violations.append("gate result must pass truthful no-op PRs without running review")

    manual_on = _workflow_on(manual_data)
    if "pull_request" in manual_on:
        violations.append("manual Aragora review workflow must not trigger on pull_request")

    return violations


def check_repo(repo_root: Path) -> list[Violation]:
    gate_path = repo_root / REVIEW_GATE_WORKFLOW
    manual_path = repo_root / MANUAL_REVIEW_WORKFLOW
    missing = []
    for path in (gate_path, manual_path):
        if not path.exists():
            missing.append(
                Violation(path=str(path.relative_to(repo_root)), message="missing workflow file")
            )
    if missing:
        return missing

    gate_data, gate_text = _load_workflow(gate_path)
    manual_data, _ = _load_workflow(manual_path)
    return [
        Violation(path=str(REVIEW_GATE_WORKFLOW), message=message)
        for message in find_review_gate_policy_violations(gate_data, gate_text, manual_data)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce Aragora PR review gate workflow policy.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to check",
    )
    args = parser.parse_args()

    violations = check_repo(Path(args.repo_root).resolve())
    if not violations:
        print("Aragora review gate policy check passed")
        return 0

    print("Aragora review gate policy violations detected:")
    for violation in violations:
        print(f"- {violation.path}: {violation.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

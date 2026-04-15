#!/usr/bin/env python3
"""Enforce checkout-integrity guards for self-hosted workflow jobs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


WORKFLOW_ROOT = Path(".github/workflows")
WORKFLOW_GLOBS = ("*.yml", "*.yaml")
CHECKOUT_ACTION_PATH = Path(".github/actions/checkout-integrity/action.yml")
ARCHIVE_RESTORE_SNIPPET = 'git archive "${GITHUB_SHA:-HEAD}" | tar -x || git archive HEAD | tar -x'
CHECKOUT_PATH_INPUT_SNIPPET = "checkout-path:"
ROOT_GUARD_USES = "./.github/actions/checkout-integrity"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    message: str


def _iter_workflow_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    workflow_dir = repo_root / WORKFLOW_ROOT
    if not workflow_dir.exists():
        return files
    for pattern in WORKFLOW_GLOBS:
        files.extend(sorted(workflow_dir.rglob(pattern)))
    return sorted(set(files))


def _line_and_column(text: str, needle: str) -> tuple[int, int]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        column = line.find(needle)
        if column != -1:
            return line_number, column + 1
    return 1, 1


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return data if isinstance(data, dict) else {}


def _is_self_hosted(runs_on: Any) -> bool:
    if isinstance(runs_on, str):
        return runs_on in {"aragora", "self-hosted"}
    if isinstance(runs_on, list):
        labels = {str(item) for item in runs_on}
        return "aragora" in labels or "self-hosted" in labels
    return False


def _is_checkout_step(step: dict[str, Any]) -> bool:
    return str(step.get("uses", "")).startswith("actions/checkout@")


def _expected_guard_uses(checkout_path: str) -> str:
    if checkout_path == ".":
        return ROOT_GUARD_USES
    return f"./{checkout_path}/.github/actions/checkout-integrity"


def _checkout_path(step: dict[str, Any]) -> str:
    with_config = step.get("with")
    if isinstance(with_config, dict):
        value = str(with_config.get("path", ".")).strip()
        return value or "."
    return "."


def _inline_guard_has_archive_restore(step: dict[str, Any]) -> bool:
    return str(
        step.get("name", "")
    ) == "Verify checkout integrity" and ARCHIVE_RESTORE_SNIPPET in str(step.get("run", ""))


def _shared_guard_matches_path(step: dict[str, Any], checkout_path: str) -> bool:
    uses = str(step.get("uses", "")).strip()
    if uses != _expected_guard_uses(checkout_path):
        return False
    with_config = step.get("with")
    configured_path = "."
    if isinstance(with_config, dict):
        configured_path = str(with_config.get("checkout-path", ".")).strip() or "."
    return configured_path == checkout_path


def check_repo(repo_root: Path) -> list[Violation]:
    violations: list[Violation] = []

    action_path = repo_root / CHECKOUT_ACTION_PATH
    if not action_path.exists():
        return [
            Violation(
                path=str(CHECKOUT_ACTION_PATH),
                line=1,
                column=1,
                message="checkout-integrity action file is missing",
            )
        ]

    action_text = action_path.read_text(encoding="utf-8")
    if ARCHIVE_RESTORE_SNIPPET not in action_text:
        line, column = _line_and_column(action_text, "name: Verify checkout integrity")
        violations.append(
            Violation(
                path=str(CHECKOUT_ACTION_PATH),
                line=line,
                column=column,
                message="checkout-integrity action must include archive restore fallback",
            )
        )
    if CHECKOUT_PATH_INPUT_SNIPPET not in action_text:
        line, column = _line_and_column(action_text, "inputs:")
        violations.append(
            Violation(
                path=str(CHECKOUT_ACTION_PATH),
                line=line,
                column=column,
                message="checkout-integrity action must support checkout-path inputs",
            )
        )

    workflow_files = _iter_workflow_files(repo_root)
    if not workflow_files:
        violations.append(
            Violation(
                path=str(WORKFLOW_ROOT),
                line=1,
                column=1,
                message="workflow directory not found or empty",
            )
        )
        return violations

    for workflow_file in workflow_files:
        text = workflow_file.read_text(encoding="utf-8")
        rel = workflow_file.relative_to(repo_root)
        data = _load_yaml_mapping(workflow_file)
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            continue

        for job_name, job in jobs.items():
            if not isinstance(job, dict) or not _is_self_hosted(job.get("runs-on")):
                continue
            steps = job.get("steps")
            if not isinstance(steps, list):
                continue

            typed_steps = [step for step in steps if isinstance(step, dict)]
            for index, step in enumerate(typed_steps):
                if not _is_checkout_step(step):
                    continue

                checkout_path = _checkout_path(step)
                later_steps = typed_steps[index + 1 :]
                if checkout_path == ".":
                    guard_ok = any(
                        _inline_guard_has_archive_restore(candidate)
                        or _shared_guard_matches_path(candidate, checkout_path)
                        for candidate in later_steps
                    )
                    message = (
                        f"self-hosted job '{job_name}' checks out the repo without a checkout-integrity "
                        "guard after actions/checkout"
                    )
                else:
                    guard_ok = any(
                        _shared_guard_matches_path(candidate, checkout_path)
                        for candidate in later_steps
                    )
                    message = (
                        f"self-hosted job '{job_name}' checks out into '{checkout_path}' but does not use "
                        f"{_expected_guard_uses(checkout_path)} with checkout-path={checkout_path}"
                    )

                if guard_ok:
                    continue

                line, column = _line_and_column(text, f"{job_name}:")
                violations.append(
                    Violation(path=str(rel), line=line, column=column, message=message)
                )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce checkout-integrity guards in self-hosted workflow jobs."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to check",
    )
    args = parser.parse_args()

    violations = check_repo(Path(args.repo_root).resolve())
    if not violations:
        print("Workflow checkout-integrity policy check passed")
        return 0

    print("Workflow checkout-integrity policy violations detected:")
    for violation in violations:
        print(f"- {violation.path}:{violation.line}:{violation.column}: {violation.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Guard required-check-priority keep-lists against drift."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Violation:
    path: str
    message: str


WORKFLOW_PATH = Path(".github/workflows/required-check-priority.yml")

REQUIRED_KEEP_WORKFLOW_PATHS = {
    ".github/workflows/lint.yml",
    ".github/workflows/sdk-parity.yml",
    ".github/workflows/sdk-test.yml",
    ".github/workflows/openapi.yml",
    ".github/workflows/required-check-priority.yml",
}

REQUIRED_KEEP_WORKFLOW_NAMES = {
    "Required Check Priority",
    "Lint",
    "SDK Parity Check",
    "SDK Tests",
    "OpenAPI Spec",
}

REQUIRED_CONTEXT_TO_WORKFLOW_PATH = {
    "lint": ".github/workflows/lint.yml",
    "typecheck": ".github/workflows/lint.yml",
    "sdk-parity": ".github/workflows/sdk-parity.yml",
    "Generate & Validate": ".github/workflows/openapi.yml",
    "TypeScript SDK Type Check": ".github/workflows/sdk-test.yml",
}


def _extract_js_set_items(workflow_text: str, set_name: str) -> list[str] | None:
    pattern = r"const\s+" + re.escape(set_name) + r"\s*=\s*new Set\(\[(?P<body>.*?)\]\);"
    match = re.search(pattern, workflow_text, flags=re.DOTALL)
    if not match:
        return None
    body = match.group("body")
    return re.findall(r"""["']([^"']+)["']""", body)


def find_required_check_priority_violations(
    workflow_text: str,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    violations: list[str] = []

    path_items = _extract_js_set_items(workflow_text, "alwaysKeepWorkflowPaths")
    if path_items is None:
        return ["missing `alwaysKeepWorkflowPaths` set definition"]

    name_items = _extract_js_set_items(workflow_text, "alwaysKeepWorkflowNames")
    if name_items is None:
        return ["missing `alwaysKeepWorkflowNames` set definition"]

    if len(path_items) != len(set(path_items)):
        violations.append("duplicate entries found in alwaysKeepWorkflowPaths")
    if len(name_items) != len(set(name_items)):
        violations.append("duplicate entries found in alwaysKeepWorkflowNames")

    path_set = set(path_items)
    missing_required_paths = sorted(REQUIRED_KEEP_WORKFLOW_PATHS - path_set)
    for path in missing_required_paths:
        violations.append(f"missing required keep workflow path: {path}")

    for context, mapped_path in sorted(REQUIRED_CONTEXT_TO_WORKFLOW_PATH.items()):
        if mapped_path not in path_set:
            violations.append(
                f"required context `{context}` maps to workflow path not in keep-list: {mapped_path}"
            )

    name_set = set(name_items)
    missing_required_names = sorted(REQUIRED_KEEP_WORKFLOW_NAMES - name_set)
    for name in missing_required_names:
        violations.append(f"missing required keep workflow name: {name}")

    if repo_root is not None:
        for rel in sorted(path_set):
            wf_path = (repo_root / rel).resolve()
            if not wf_path.exists():
                violations.append(f"keep workflow path does not exist: {rel}")
        for context, rel in sorted(REQUIRED_CONTEXT_TO_WORKFLOW_PATH.items()):
            wf_path = (repo_root / rel).resolve()
            if not wf_path.exists():
                continue
            text = wf_path.read_text(encoding="utf-8")
            if context not in text:
                violations.append(
                    f"required context marker `{context}` not found in mapped workflow: {rel}"
                )

    return violations


def check_repo(repo_root: Path) -> list[Violation]:
    workflow_file = repo_root / WORKFLOW_PATH
    if not workflow_file.exists():
        return [Violation(path=str(WORKFLOW_PATH), message="missing workflow file")]

    text = workflow_file.read_text(encoding="utf-8")
    return [
        Violation(path=str(WORKFLOW_PATH), message=message)
        for message in find_required_check_priority_violations(text, repo_root=repo_root)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce required-check-priority workflow keep-list policy."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to check",
    )
    args = parser.parse_args()

    violations = check_repo(Path(args.repo_root).resolve())
    if not violations:
        print("Required check priority policy check passed")
        return 0

    print("Required check priority policy violations detected:")
    for v in violations:
        print(f"- {v.path}: {v.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

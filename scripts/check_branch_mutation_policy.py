#!/usr/bin/env python3
"""Guard against CI/session defaults that silently mutate active PR branches.

This script enforces two policies:
1) Worktree/autopilot defaults must use non-mutating `ff-only` integration.
2) Workflow files that run `git push` must be explicitly allowlisted and constrained.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    path: str
    message: str


# path -> list[regex patterns that must match]
STRATEGY_REQUIRED_PATTERNS: dict[str, list[str]] = {
    "scripts/codex_session.sh": [r"--reconcile --strategy ff-only"],
    "scripts/install_worktree_maintainer_launchd.sh": [
        r'^STRATEGY="ff-only"$',
        r"Integration strategy \(default: ff-only\)",
    ],
    "aragora/worktree/autopilot.py": [r'strategy:\s*str\s*=\s*"ff-only"'],
    "aragora/worktree/maintainer.py": [r'default="ff-only"'],
    # ensure_managed_worktree + maintain_managed_dirs defaults
    "aragora/worktree/lifecycle.py": [r'strategy:\s*str\s*=\s*"ff-only"'],
    # ensure + reconcile + maintain parser defaults
    "scripts/codex_worktree_autopilot.py": [r'default="ff-only"'],
}

REQUIRED_MATCH_COUNTS: dict[str, int] = {
    "aragora/worktree/lifecycle.py": 2,
    "scripts/codex_worktree_autopilot.py": 3,
}

WORKFLOW_MUTATION_ALLOWLIST = {
    "openapi.yml",
    "release-notes.yml",
    "testfixer-auto.yml",
}


def find_strategy_default_violations(files: dict[str, str]) -> list[Violation]:
    violations: list[Violation] = []
    for path, patterns in STRATEGY_REQUIRED_PATTERNS.items():
        text = files.get(path)
        if text is None:
            violations.append(Violation(path=path, message="missing required file"))
            continue

        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.MULTILINE)
            if not matches:
                violations.append(
                    Violation(path=path, message=f"missing required pattern: {pattern}")
                )
            total_matches += len(matches)

        min_count = REQUIRED_MATCH_COUNTS.get(path, len(patterns))
        if total_matches < min_count:
            violations.append(
                Violation(
                    path=path,
                    message=(
                        f"expected at least {min_count} ff-only default markers, "
                        f"found {total_matches}"
                    ),
                )
            )
    return violations


def find_mutating_workflow_violations(workflows: dict[str, str]) -> list[Violation]:
    violations: list[Violation] = []
    for name, text in workflows.items():
        if "git push" not in text:
            continue

        if name not in WORKFLOW_MUTATION_ALLOWLIST:
            violations.append(
                Violation(
                    path=f".github/workflows/{name}",
                    message=(
                        "contains `git push` but is not in mutation allowlist "
                        f"{sorted(WORKFLOW_MUTATION_ALLOWLIST)}"
                    ),
                )
            )
            continue

        if name == "openapi.yml":
            if "if: github.event_name == 'push' && github.ref == 'refs/heads/main'" not in text:
                violations.append(
                    Violation(
                        path=f".github/workflows/{name}",
                        message="must gate git push behind push-to-main condition",
                    )
                )

        if name == "release-notes.yml":
            if re.search(r"^\s*pull_request(_target)?\s*:", text, flags=re.MULTILINE):
                violations.append(
                    Violation(
                        path=f".github/workflows/{name}",
                        message="must not be triggered by pull_request/pull_request_target",
                    )
                )

        if name == "testfixer-auto.yml":
            if not re.search(r'branch="testfixer/', text):
                violations.append(
                    Violation(
                        path=f".github/workflows/{name}",
                        message="must push only to testfixer/* branch namespace",
                    )
                )
    return violations


def _load_repo_files(repo_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    strategy_files: dict[str, str] = {}
    for rel in STRATEGY_REQUIRED_PATTERNS:
        path = repo_root / rel
        if path.exists():
            strategy_files[rel] = path.read_text(encoding="utf-8")

    workflows: dict[str, str] = {}
    workflows_dir = repo_root / ".github" / "workflows"
    if workflows_dir.exists():
        for wf in sorted(workflows_dir.glob("*.yml")):
            workflows[wf.name] = wf.read_text(encoding="utf-8")
    return strategy_files, workflows


def check_repo(repo_root: Path) -> list[Violation]:
    strategy_files, workflows = _load_repo_files(repo_root)
    violations = []
    violations.extend(find_strategy_default_violations(strategy_files))
    violations.extend(find_mutating_workflow_violations(workflows))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce branch mutation safety defaults.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to check",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    violations = check_repo(repo_root)
    if not violations:
        print("Branch mutation policy check passed")
        return 0

    print("Branch mutation policy violations detected:")
    for v in violations:
        print(f"- {v.path}: {v.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

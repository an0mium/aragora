#!/usr/bin/env python3
"""Guard actions/github-script@v9 script blocks against known v9 breaks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    message: str


GITHUB_SCRIPT_V9_RE = re.compile(r"\buses:\s*['\"]?actions/github-script@v9(?:['\"]|\s|$)")
SCRIPT_BLOCK_RE = re.compile(r"^\s*script:\s*[|>]")
GET_OCTOKIT_REDECLARATION_RE = re.compile(r"\b(?:const|let)\s+(?:\{\s*)?getOctokit\b")
DIRECT_ACTIONS_GITHUB_REQUIRE_RE = re.compile(r"\brequire\(\s*['\"]@actions/github['\"]\s*\)")
ACTIONS_GITHUB_INTERNAL_RE = re.compile(r"['\"]@actions/github/[^'\"]+['\"]")
WORKFLOW_ROOTS = (Path(".github/workflows"), Path(".github/actions"))
WORKFLOW_SUFFIXES = {".yml", ".yaml"}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _violations_for_script_line(line: str) -> list[str]:
    messages: list[str] = []
    if GET_OCTOKIT_REDECLARATION_RE.search(line):
        messages.append("do not redeclare injected getOctokit in actions/github-script@v9 scripts")
    if ACTIONS_GITHUB_INTERNAL_RE.search(line):
        messages.append("do not use @actions/github internals in actions/github-script@v9 scripts")
    elif DIRECT_ACTIONS_GITHUB_REQUIRE_RE.search(line):
        messages.append(
            "require('@actions/github') fails under actions/github-script@v9; "
            "use the injected github or getOctokit context instead"
        )
    return messages


def find_github_script_v9_violations(workflow_text: str) -> list[tuple[int, str, str]]:
    lines = workflow_text.splitlines()
    violations: list[tuple[int, str, str]] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not GITHUB_SCRIPT_V9_RE.search(line):
            index += 1
            continue

        step_indent = _indent(line)
        index += 1
        while index < len(lines):
            current = lines[index]
            stripped = current.strip()
            if stripped and _indent(current) <= step_indent:
                break

            if not SCRIPT_BLOCK_RE.match(current):
                index += 1
                continue

            script_indent = _indent(current)
            index += 1
            while index < len(lines):
                script_line = lines[index]
                if script_line.strip() and _indent(script_line) <= script_indent:
                    break

                for message in _violations_for_script_line(script_line):
                    violations.append((index + 1, script_line.strip(), message))
                index += 1

    return violations


def _iter_workflow_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root in WORKFLOW_ROOTS:
        full_root = repo_root / root
        if not full_root.exists():
            continue
        files.extend(
            path
            for path in full_root.rglob("*")
            if path.is_file() and path.suffix in WORKFLOW_SUFFIXES
        )
    return sorted(files)


def check_repo(repo_root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for workflow_file in _iter_workflow_files(repo_root):
        text = workflow_file.read_text(encoding="utf-8")
        rel_path = str(workflow_file.relative_to(repo_root))
        violations.extend(
            Violation(path=rel_path, line=line, message=message)
            for line, _snippet, message in find_github_script_v9_violations(text)
        )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce actions/github-script@v9 script compatibility policy."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to check",
    )
    args = parser.parse_args()

    violations = check_repo(Path(args.repo_root).resolve())
    if not violations:
        print("github-script v9 policy check passed")
        return 0

    print("github-script v9 policy violations detected:")
    for violation in violations:
        print(f"- {violation.path}:{violation.line}: {violation.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

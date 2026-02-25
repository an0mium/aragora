#!/usr/bin/env python3
"""Fail CI when legacy Codex collab flags/messages leak into repo-facing surfaces."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = [
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    ".github/workflows",
    "docs",
    "docs-site/docs",
    "scripts",
    "deploy",
]

INCLUDE_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".yaml",
    ".yml",
    ".txt",
    ".toml",
    ".ini",
    ".cfg",
}

EXCLUDED_PREFIXES = (
    ".git/",
    "docs-site/build/",
    "deprecated/",
    "node_modules/",
    "aragora/live/docs/api/",
)

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "legacy_codex_collab_flag",
        re.compile(r"\bcodex(?:\s+\S+)*\s+--collab(?:\s|$)", re.IGNORECASE),
    ),
    (
        "legacy_codex_enable_multi_agent_flag",
        re.compile(r"\bcodex(?:\s+\S+)*\s+--enable\s+multi_agent(?:\s|$)", re.IGNORECASE),
    ),
    (
        "legacy_collab_feature_toggle",
        re.compile(r"\bcollab\s*=\s*true\b", re.IGNORECASE),
    ),
]


def _iter_files(root: Path, rel_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for rel in rel_paths:
        path = (root / rel).resolve()
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            rel_posix = candidate.relative_to(root).as_posix()
            if rel_posix.startswith(EXCLUDED_PREFIXES):
                continue
            if candidate.suffix.lower() not in INCLUDE_SUFFIXES:
                continue
            files.append(candidate)
    return files


def _scan_file(file_path: Path, root: Path) -> list[tuple[str, int, str, str]]:
    issues: list[tuple[str, int, str, str]] = []
    rel = file_path.relative_to(root).as_posix()
    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return issues

    for line_no, line in enumerate(lines, start=1):
        for code, regex in PATTERNS:
            if regex.search(line):
                issues.append((rel, line_no, code, line.strip()))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for legacy Codex collab flags")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_PATHS,
        help="Files/directories to scan (relative to root)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = _iter_files(root, args.paths)

    issues: list[tuple[str, int, str, str]] = []
    for file_path in files:
        issues.extend(_scan_file(file_path, root))

    if issues:
        print("Legacy Codex collab patterns found:", file=sys.stderr)
        for rel, line_no, code, snippet in issues:
            print(f"- {rel}:{line_no} [{code}] {snippet}", file=sys.stderr)
            print(f"::error file={rel},line={line_no}::{code}: {snippet}")
        print(
            "\nRemove legacy collab references and rely on current Codex defaults/config.",
            file=sys.stderr,
        )
        return 1

    print("No legacy Codex collab patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

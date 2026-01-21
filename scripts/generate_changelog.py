#!/usr/bin/env python3
"""
Generate changelog from git commits following Conventional Commits.

Parses commit messages and generates a structured changelog grouped by type.

Usage:
    python scripts/generate_changelog.py --from v2.1.0 --to v2.2.0
    python scripts/generate_changelog.py --from HEAD~50 --to HEAD

Conventional Commits Format:
    type(scope): description

Types:
    - feat: New features
    - fix: Bug fixes
    - perf: Performance improvements
    - refactor: Code refactoring
    - docs: Documentation
    - test: Tests
    - chore: Maintenance
    - ci: CI/CD changes
    - build: Build system
    - style: Code style
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Commit:
    """Represents a parsed commit."""

    hash: str
    type: str
    scope: Optional[str]
    description: str
    body: str
    breaking: bool
    author: str
    date: str


# Conventional commit type labels
TYPE_LABELS = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "perf": "Performance Improvements",
    "refactor": "Code Refactoring",
    "docs": "Documentation",
    "test": "Tests",
    "chore": "Maintenance",
    "ci": "CI/CD",
    "build": "Build System",
    "style": "Code Style",
    "revert": "Reverts",
}

# Type display order
TYPE_ORDER = [
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "test",
    "chore",
    "ci",
    "build",
    "style",
    "revert",
]

# Regex for conventional commits
COMMIT_PATTERN = re.compile(
    r"^(?P<type>\w+)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?P<breaking>!)?"
    r":\s*"
    r"(?P<description>.+)$"
)


def get_commits(from_ref: str, to_ref: str) -> List[str]:
    """Get commit messages between two refs."""
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                f"{from_ref}..{to_ref}",
                "--format=%H|%an|%ad|%s|%b%x00",
                "--date=short",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return [c.strip() for c in result.stdout.split("\x00") if c.strip()]
    except subprocess.CalledProcessError as e:
        print(f"Error getting commits: {e}", file=sys.stderr)
        return []


def parse_commit(raw: str) -> Optional[Commit]:
    """Parse a raw commit string into a Commit object."""
    parts = raw.split("|", 4)
    if len(parts) < 4:
        return None

    commit_hash = parts[0]
    author = parts[1]
    date = parts[2]
    subject = parts[3]
    body = parts[4] if len(parts) > 4 else ""

    # Parse conventional commit format
    match = COMMIT_PATTERN.match(subject)
    if not match:
        # Non-conventional commit, categorize as "chore"
        return Commit(
            hash=commit_hash,
            type="chore",
            scope=None,
            description=subject,
            body=body,
            breaking=False,
            author=author,
            date=date,
        )

    commit_type = match.group("type").lower()
    scope = match.group("scope")
    breaking = match.group("breaking") == "!" or "BREAKING CHANGE" in body
    description = match.group("description")

    return Commit(
        hash=commit_hash,
        type=commit_type if commit_type in TYPE_LABELS else "chore",
        scope=scope,
        description=description,
        body=body,
        breaking=breaking,
        author=author,
        date=date,
    )


def generate_changelog(
    commits: List[Commit],
    version: str,
    date: Optional[str] = None,
) -> str:
    """Generate markdown changelog from commits."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Group commits by type
    grouped: Dict[str, List[Commit]] = defaultdict(list)
    breaking_changes: List[Commit] = []

    for commit in commits:
        grouped[commit.type].append(commit)
        if commit.breaking:
            breaking_changes.append(commit)

    # Build changelog
    lines = [
        f"## [{version}] - {date}",
        "",
    ]

    # Breaking changes section
    if breaking_changes:
        lines.extend(
            [
                "### Breaking Changes",
                "",
            ]
        )
        for commit in breaking_changes:
            scope_str = f"**{commit.scope}:** " if commit.scope else ""
            lines.append(f"- {scope_str}{commit.description}")
        lines.append("")

    # Regular sections by type
    for commit_type in TYPE_ORDER:
        type_commits = grouped.get(commit_type, [])
        if not type_commits:
            continue

        label = TYPE_LABELS.get(commit_type, commit_type.title())
        lines.extend(
            [
                f"### {label}",
                "",
            ]
        )

        # Group by scope within type
        scoped: Dict[Optional[str], List[Commit]] = defaultdict(list)
        for commit in type_commits:
            scoped[commit.scope].append(commit)

        for scope in sorted(scoped.keys(), key=lambda x: x or ""):
            scope_commits = scoped[scope]
            for commit in scope_commits:
                scope_str = f"**{scope}:** " if scope else ""
                short_hash = commit.hash[:7]
                lines.append(f"- {scope_str}{commit.description} ({short_hash})")

        lines.append("")

    # Contributors
    authors = sorted(set(c.author for c in commits))
    if authors:
        lines.extend(
            [
                "### Contributors",
                "",
            ]
        )
        for author in authors:
            lines.append(f"- {author}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate changelog from git commits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from",
        dest="from_ref",
        required=True,
        help="Starting git ref (tag, commit, branch)",
    )
    parser.add_argument(
        "--to",
        dest="to_ref",
        default="HEAD",
        help="Ending git ref (default: HEAD)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--version",
        "-v",
        help="Version string (default: derived from --to)",
    )

    args = parser.parse_args()

    # Get version
    version = args.version
    if not version:
        if args.to_ref.startswith("v"):
            version = args.to_ref
        elif args.to_ref == "HEAD":
            version = "Unreleased"
        else:
            version = args.to_ref[:7]

    # Get and parse commits
    raw_commits = get_commits(args.from_ref, args.to_ref)
    commits = []
    for raw in raw_commits:
        commit = parse_commit(raw)
        if commit:
            commits.append(commit)

    if not commits:
        print("No commits found", file=sys.stderr)
        sys.exit(1)

    # Generate changelog
    changelog = generate_changelog(commits, version)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(changelog)
        print(f"Changelog written to {args.output}")
    else:
        print(changelog)


if __name__ == "__main__":
    main()

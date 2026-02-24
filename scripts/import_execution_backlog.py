#!/usr/bin/env python3
"""Import execution milestones and backlog issues into GitHub.

Usage:
  python scripts/import_execution_backlog.py
  python scripts/import_execution_backlog.py --apply
  python scripts/import_execution_backlog.py --apply --repo owner/repo

Default mode is dry-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def detect_repo(explicit_repo: str | None) -> str:
    if explicit_repo:
        return explicit_repo

    result = run(["git", "config", "--get", "remote.origin.url"], check=False)
    remote = (result.stdout or "").strip()
    if not remote:
        raise RuntimeError("Could not detect remote.origin.url; pass --repo owner/repo")

    patterns = [
        r"github\.com[:/](?P<repo>[^/]+/[^/.]+)(?:\.git)?$",
        r"^(?P<repo>[^/]+/[^/]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, remote)
        if match:
            return match.group("repo")

    raise RuntimeError(f"Unsupported remote format: {remote}")


def load_milestones(path: Path) -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = (row.get("title") or "").strip()
            due_on = (row.get("due_on") or "").strip()
            description = (row.get("description") or "").strip()
            if not title:
                continue
            data[title] = {"due_on": due_on, "description": description}
    return data


def load_issues(path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = (row.get("title") or "").strip()
            if not title:
                continue
            items.append(
                {
                    "title": title,
                    "body": (row.get("body") or "").strip(),
                    "labels": (row.get("labels") or "").strip(),
                    "milestone": (row.get("milestone") or "").strip(),
                }
            )
    return items


def get_existing_milestones(repo: str) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    page = 1
    while True:
        result = run(
            [
                "gh",
                "api",
                "--method",
                "GET",
                f"repos/{repo}/milestones",
                "-f",
                "state=open",
                "-F",
                f"page={page}",
                "-F",
                "per_page=100",
            ],
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Failed to list milestones: {stderr}")

        rows = json.loads(result.stdout or "[]")
        if not rows:
            break
        for row in rows:
            title = row.get("title")
            if title:
                existing[title] = row
        if len(rows) < 100:
            break
        page += 1

    return existing


def milestone_issue_exists(repo: str, title: str) -> bool:
    # Use exact title search bounded to repo, returning first page only for speed.
    query = f'repo:{repo} is:issue in:title "{title}"'
    result = run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--search",
            query,
            "--limit",
            "20",
            "--json",
            "title",
        ],
        check=False,
    )
    if result.returncode != 0:
        return False

    rows = json.loads(result.stdout or "[]")
    return any((row.get("title") or "").strip() == title for row in rows)


def ensure_milestones(
    repo: str,
    milestones: dict[str, dict[str, str]],
    apply: bool,
) -> None:
    if not milestones:
        return

    if not apply:
        print("[dry-run] milestones to ensure:")
        for title, payload in milestones.items():
            print(f"  - {title} (due_on={payload.get('due_on')})")
        return

    existing = get_existing_milestones(repo)
    for title, payload in milestones.items():
        if title in existing:
            print(f"[skip] milestone exists: {title}")
            continue

        cmd = ["gh", "api", "-X", "POST", f"repos/{repo}/milestones", "-f", f"title={title}"]
        if payload.get("description"):
            cmd.extend(["-f", f"description={payload['description']}"])
        due_on = payload.get("due_on")
        if due_on:
            cmd.extend(["-f", f"due_on={due_on}T00:00:00Z"])

        result = run(cmd, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Failed to create milestone '{title}': {stderr}")
        print(f"[create] milestone: {title}")


def create_issues(
    repo: str,
    issues: list[dict[str, str]],
    apply: bool,
    skip_existing: bool,
) -> None:
    for issue in issues:
        title = issue["title"]
        labels = [label.strip() for label in issue["labels"].split(",") if label.strip()]
        milestone = issue["milestone"]
        body = issue["body"]

        if not apply:
            print(f"[dry-run] issue: {title}")
            continue

        if skip_existing and milestone_issue_exists(repo, title):
            print(f"[skip] issue exists: {title}")
            continue

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(body)
            body_file = handle.name

        cmd = [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            title,
            "--body-file",
            body_file,
        ]
        for label in labels:
            cmd.extend(["--label", label])
        if milestone:
            cmd.extend(["--milestone", milestone])

        result = run(cmd, check=False)
        Path(body_file).unlink(missing_ok=True)

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Failed to create issue '{title}': {stderr}")

        url = (result.stdout or "").strip()
        print(f"[create] issue: {title} -> {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import execution backlog into GitHub.")
    parser.add_argument(
        "--repo",
        help="GitHub repo in owner/name format (auto-detected from git remote if omitted).",
    )
    parser.add_argument(
        "--milestones-csv",
        default="docs/status/EXECUTION_MILESTONES_2026Q2.csv",
        help="Path to milestones CSV.",
    )
    parser.add_argument(
        "--issues-csv",
        default="docs/status/EXECUTION_BACKLOG_2026Q2.csv",
        help="Path to issues CSV.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create milestones and issues. Default is dry-run.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not check for duplicate issue titles before creating.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        repo = detect_repo(args.repo)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    milestones_path = Path(args.milestones_csv)
    issues_path = Path(args.issues_csv)

    if not milestones_path.exists():
        print(f"error: milestones CSV not found: {milestones_path}", file=sys.stderr)
        return 2
    if not issues_path.exists():
        print(f"error: issues CSV not found: {issues_path}", file=sys.stderr)
        return 2

    milestones = load_milestones(milestones_path)
    issues = load_issues(issues_path)

    mode = "apply" if args.apply else "dry-run"
    print(f"mode={mode} repo={repo} milestones={len(milestones)} issues={len(issues)}")

    try:
        ensure_milestones(repo=repo, milestones=milestones, apply=args.apply)
        create_issues(
            repo=repo,
            issues=issues,
            apply=args.apply,
            skip_existing=not args.no_skip_existing,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

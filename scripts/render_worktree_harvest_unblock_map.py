#!/usr/bin/env python3
"""Render a read-only unblock map from Codex worktree-harvest inventory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path(".aragora/worktree-harvest/latest.json")
DEFAULT_LIMIT = 25
MUTATION_WARNING_PREFIX = "DO NOT RUN: "


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _size_bytes(candidate: dict[str, Any]) -> int:
    try:
        return int(candidate.get("size_bytes") or 0)
    except (TypeError, ValueError):
        return 0


def _size_gib(size_bytes: int) -> float:
    return round(size_bytes / 1024**3, 3)


def _candidate_git(candidate: dict[str, Any]) -> dict[str, Any]:
    git = candidate.get("git")
    return git if isinstance(git, dict) else {}


def _candidate_links(candidate: dict[str, Any]) -> dict[str, Any]:
    links = candidate.get("links")
    return links if isinstance(links, dict) else {}


def blocker_family(candidate: dict[str, Any]) -> str:
    classification = str(candidate.get("classification") or "")
    git = _candidate_git(candidate)
    links = _candidate_links(candidate)
    lock_files = candidate.get("lock_files")

    if candidate.get("cleanup_candidate") is True:
        return "cleanup candidate"
    if classification == "unique_unharvested" or int(git.get("ahead") or 0) > 0:
        return "unique commits not on origin/main"
    if (
        classification == "receipt_protected"
        or links.get("receipt_files")
        or links.get("outbox_files")
    ):
        return "receipt/outbox protected"
    if classification == "open_pr_or_outbox" or links.get("open_prs"):
        return "open PR protected"
    if classification == "lookup_failed" or git.get("lookup_failed"):
        return "helper parse/metadata ambiguity"
    if git.get("dirty"):
        return "dirty uncommitted changes"
    if candidate.get("active_session") or lock_files:
        return "active session lock"
    if classification in {
        "patch_equivalent_or_merged",
        "unregistered_git_residue",
        "no_git_cache_residue",
    }:
        return "cleanup candidate"
    return classification or "unknown"


def _candidate_row(candidate: dict[str, Any]) -> dict[str, Any]:
    git = _candidate_git(candidate)
    family = blocker_family(candidate)
    path = str(candidate.get("path") or "")
    repo_path = str(candidate.get("repo_path") or path)
    if candidate.get("cleanup_candidate") is True:
        next_command = (
            f"{MUTATION_WARNING_PREFIX}python3 scripts/safe_worktree_cleanup.py "
            f"inspect {json.dumps(path)} --json"
        )
    elif family == "dirty uncommitted changes":
        next_command = f"{MUTATION_WARNING_PREFIX}git -C {json.dumps(repo_path)} status --short"
    elif family == "unique commits not on origin/main":
        next_command = f"{MUTATION_WARNING_PREFIX}git -C {json.dumps(repo_path)} log --oneline origin/main..HEAD"
    else:
        next_command = f"{MUTATION_WARNING_PREFIX}python3 scripts/safe_worktree_cleanup.py inspect {json.dumps(path)} --json"
    size_bytes = _size_bytes(candidate)
    return {
        "path": path,
        "repo_path": candidate.get("repo_path"),
        "classification": candidate.get("classification"),
        "blocker_family": family,
        "size_bytes": size_bytes,
        "size_gib": _size_gib(size_bytes),
        "branch": git.get("branch"),
        "head": git.get("head"),
        "dirty": bool(git.get("dirty")),
        "ahead": git.get("ahead"),
        "proof": list(candidate.get("proof") or []),
        "next_command": next_command,
    }


def _top(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=_size_bytes, reverse=True)
    return [_candidate_row(candidate) for candidate in ordered[:limit]]


def render_unblock_map(inventory: dict[str, Any], *, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    candidates = [
        candidate for candidate in inventory.get("candidates", []) if isinstance(candidate, dict)
    ]
    raw_summary = inventory.get("summary")
    summary: dict[str, Any] = raw_summary if isinstance(raw_summary, dict) else {}
    family_counts: Counter[str] = Counter()
    family_bytes: defaultdict[str, int] = defaultdict(int)
    for candidate in candidates:
        family = blocker_family(candidate)
        family_counts[family] += 1
        family_bytes[family] += _size_bytes(candidate)

    blocker_families = [
        {
            "family": family,
            "count": count,
            "size_bytes": family_bytes[family],
            "size_gib": _size_gib(family_bytes[family]),
        }
        for family, count in family_counts.most_common()
    ]

    cleanup_candidates = [
        candidate for candidate in candidates if candidate.get("cleanup_candidate") is True
    ]
    human_review_candidates = [
        candidate
        for candidate in candidates
        if blocker_family(candidate)
        in {"dirty uncommitted changes", "unique commits not on origin/main"}
    ]
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "source_inventory": {
            "generated_at": inventory.get("generated_at"),
            "root": inventory.get("root"),
            "schema": inventory.get("schema"),
        },
        "summary": {
            "total_candidates": summary.get("total_candidates", len(candidates)),
            "cleanup_candidate_count": summary.get(
                "cleanup_candidate_count", len(cleanup_candidates)
            ),
            "harvest_candidate_count": summary.get("harvest_candidate_count"),
            "known_size_bytes": summary.get("known_size_bytes"),
            "known_size_gib": _size_gib(int(summary.get("known_size_bytes") or 0)),
        },
        "blocker_families": blocker_families,
        "top_cleanup_candidates": _top(cleanup_candidates, limit=limit),
        "top_human_review_candidates": _top(human_review_candidates, limit=limit),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input must be a JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a read-only unblock map from a worktree-harvest inventory."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = render_unblock_map(_load_json(args.input), limit=args.limit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

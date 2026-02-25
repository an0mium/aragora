"""CLI entrypoint for worktree maintainer operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aragora.worktree.lifecycle import WorktreeLifecycleService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintain managed codex-auto worktrees via shared lifecycle service."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("ARAGORA_REPO_ROOT", "."),
        help="Repository root (default: ARAGORA_REPO_ROOT or current directory)",
    )
    parser.add_argument("--base", default="main", help="Base branch to integrate from")
    parser.add_argument("--ttl-hours", type=int, default=24, help="Stale-session TTL in hours")
    parser.add_argument(
        "--strategy",
        choices=("merge", "rebase", "ff-only", "none"),
        default="merge",
        help="Integration strategy",
    )
    parser.add_argument(
        "--managed-dir",
        action="append",
        dest="managed_dirs",
        default=[],
        help="Managed dir relative to repo root (repeatable)",
    )
    parser.add_argument(
        "--delete-branches",
        dest="delete_branches",
        action="store_true",
        help="Allow cleanup to delete local codex/* branches",
    )
    parser.add_argument(
        "--no-delete-branches",
        dest="delete_branches",
        action="store_false",
        help="Keep local codex/* branches during cleanup",
    )
    parser.set_defaults(delete_branches=False)
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Also maintain directories with active session lock files",
    )
    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help="Only reconcile managed sessions (skip cleanup/removal phase)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    service = WorktreeLifecycleService(repo_root=Path(args.repo))
    payload = service.maintain_managed_dirs(
        base_branch=args.base,
        ttl_hours=args.ttl_hours,
        strategy=args.strategy,
        managed_dirs=args.managed_dirs or None,
        include_active=args.include_active,
        reconcile_only=args.reconcile_only,
        delete_branches=args.delete_branches,
    )

    if args.json:
        print(json.dumps(payload, indent=2))  # noqa: T201
    else:
        print(  # noqa: T201
            "worktree-maintainer: "
            f"ok={payload['ok']} total={payload['directories_total']} "
            f"processed={payload['processed']} skipped_active={payload['skipped_active']} "
            f"skipped_missing={payload['skipped_missing']} failures={payload['failures']}"
        )
        for row in payload["results"]:
            status = row.get("status", "unknown")
            managed_dir = row.get("managed_dir", "?")
            print(f"  - {managed_dir}: {status}")  # noqa: T201

    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

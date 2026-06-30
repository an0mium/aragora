#!/usr/bin/env python3
"""Build a value-preserving queue disposition manifest (READ-ONLY).

This script is a coordination artifact for queue drain runs. It classifies open
PRs and, optionally, worktree inventory candidates before any broad close/delete
batch. It never mutates GitHub or git; the only optional write is ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.queue_disposition import build_manifest  # noqa: E402

DEFAULT_REPO = "synaptent/aragora"
DEFAULT_LIMIT = 300
GH_TIMEOUT_SECONDS = 120
MERGE_PACKET_TIMEOUT_SECONDS = 60
INVENTORY_TIMEOUT_SECONDS = 300
DEFAULT_MERGE_PACKET_WORKERS = 4
MAX_MERGE_PACKET_WORKERS = 16

EXIT_OK = 0
EXIT_FAILURE = 1

_GH_FIELDS = (
    "number,title,labels,isDraft,createdAt,updatedAt,headRefName,headRefOid,"
    "baseRefName,mergeable,reviewDecision,additions,deletions,changedFiles,author"
)


def subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    return env


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def default_list_prs(repo: str, limit: int) -> list[dict[str, Any]]:
    command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--json",
        _GH_FIELDS,
        "--limit",
        str(limit),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=GH_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr list failed (exit {result.returncode}): {result.stderr.strip()}")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("gh pr list returned unexpected payload (expected a list)")
    return payload


def default_merge_packet(pr: int, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "aragora.cli.main",
        "review-queue",
        "merge-packet",
        "--pr",
        str(pr),
        "--repo",
        repo,
        "--json",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=MERGE_PACKET_TIMEOUT_SECONDS,
        check=False,
        cwd=str(REPO_ROOT),
        env=subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"merge-packet failed for PR #{pr} (exit {result.returncode}): {result.stderr.strip()}"
        )
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"merge-packet returned unexpected payload for PR #{pr}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"merge-packet returned no entries for PR #{pr}")
    entry = entries[0]
    if not isinstance(entry, dict):
        raise RuntimeError(f"merge-packet returned malformed entry for PR #{pr}")
    return entry


def default_inventory_candidates() -> list[dict[str, Any]]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "codex_worktree_value_inventory.py"),
        "--json",
        "--dry-run",
        "--include-pr-state",
        "--smart-merge-detection",
        "--size-mode",
        "none",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=INVENTORY_TIMEOUT_SECONDS,
        check=False,
        cwd=str(REPO_ROOT),
        env=subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "codex_worktree_value_inventory failed "
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )
    payload = json.loads(result.stdout or "{}")
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        raise RuntimeError("inventory returned unexpected payload (missing candidates list)")
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _merge_packet_failure_annotation(number: int, exc: BaseException) -> str:
    return f"merge_packet_failed:#{number}:{str(exc)[:200]}"


def _coerce_pr_number(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def collect_merge_packets(
    prs: list[dict[str, Any]],
    merge_packet: Callable[[int], dict[str, Any]],
    *,
    workers: int,
    annotations: list[str],
) -> dict[int, dict[str, Any]]:
    """Collect per-PR merge packets with bounded read-only parallelism."""
    jobs = [
        (pr, number) for pr in prs if (number := _coerce_pr_number(pr.get("number"))) is not None
    ]
    merge_entries: dict[int, dict[str, Any]] = {}
    if not jobs:
        return merge_entries
    workers = max(1, min(workers, len(jobs), MAX_MERGE_PACKET_WORKERS))
    if workers == 1:
        for pr, number in jobs:
            try:
                entry = merge_packet(number)
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
                annotation = _merge_packet_failure_annotation(number, exc)
                annotations.append(annotation)
                pr["_merge_packet_error"] = annotation
                continue
            if entry:
                merge_entries[number] = entry
        return merge_entries

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(merge_packet, number): (pr, number) for pr, number in jobs}
        for future in as_completed(futures):
            pr, number = futures[future]
            try:
                entry = future.result()
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
                annotation = _merge_packet_failure_annotation(number, exc)
                annotations.append(annotation)
                pr["_merge_packet_error"] = annotation
                continue
            if entry:
                merge_entries[number] = entry
    return merge_entries


def run_manifest(
    *,
    list_prs: Callable[[], list[dict[str, Any]]],
    merge_packet: Callable[[int], dict[str, Any]] | None = None,
    merge_packet_workers: int = DEFAULT_MERGE_PACKET_WORKERS,
    inventory_candidates: Callable[[], list[dict[str, Any]]] | None = None,
    limit: int = DEFAULT_LIMIT,
    stale_days: int = 14,
    out_file: str | None = None,
    summary: bool = False,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> int:
    if now is None:
        now = datetime.now(timezone.utc)
    annotations: list[str] = []
    try:
        prs = [pr for pr in list_prs() if isinstance(pr, dict)]
        if len(prs) >= limit:
            annotations.append(f"list_truncated:>={limit}")

        merge_entries: dict[int, dict[str, Any]] | None = None
        if merge_packet is not None:
            merge_entries = collect_merge_packets(
                prs,
                merge_packet,
                workers=merge_packet_workers,
                annotations=annotations,
            )

        candidates = inventory_candidates() if inventory_candidates is not None else []
        payload = build_manifest(
            prs=prs,
            merge_packet_entries=merge_entries,
            inventory_candidates=candidates,
            now=now,
            stale_days=stale_days,
            annotations=annotations,
        )
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"action": "error", "error": str(exc)[:500]}), file=sys.stderr)
        return EXIT_FAILURE

    if summary:
        counts = payload["summary"]["by_disposition"]
        log(
            "queue disposition: "
            f"total={payload['summary']['total_items']} "
            f"harvest={counts.get('harvest_now', 0)} "
            f"human={counts.get('human_packet', 0)} "
            f"park={counts.get('park_preserve', 0)} "
            f"close_delete={counts.get('close_or_delete_after_manifest', 0)} "
            f"operator_required={payload['summary']['operator_required']}"
        )
    else:
        log(json.dumps(payload, sort_keys=True))

    if out_file:
        try:
            atomic_write_json(out_file, payload)
        except OSError as exc:
            print(
                json.dumps({"action": "out_write_failed", "error": str(exc)[:300]}),
                file=sys.stderr,
            )
            return EXIT_FAILURE
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only queue disposition manifest for preserve-first drain runs."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo (owner/name)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--stale-days", type=int, default=14)
    parser.add_argument(
        "--with-merge-packets",
        dest="with_merge_packets",
        action="store_true",
        default=True,
        help="Run review-queue merge-packet for each open PR (read-only, slower).",
    )
    parser.add_argument(
        "--no-merge-packets",
        dest="with_merge_packets",
        action="store_false",
        help="Skip merge-packet collection; PR dispositions will park until packet data exists.",
    )
    parser.add_argument(
        "--merge-packet-workers",
        type=int,
        default=DEFAULT_MERGE_PACKET_WORKERS,
        help="Bounded read-only merge-packet workers when --with-merge-packets is set.",
    )
    parser.add_argument(
        "--include-worktrees",
        action="store_true",
        help="Include codex_worktree_value_inventory candidates (read-only, slower).",
    )
    parser.add_argument("--summary", action="store_true", help="Emit one summary line.")
    parser.add_argument("--out", default=None, help="Optional atomic JSON output path.")
    args = parser.parse_args(argv)
    if args.stale_days < 1:
        print(
            json.dumps({"action": "error", "error": "--stale-days must be >= 1"}), file=sys.stderr
        )
        return EXIT_FAILURE
    if not 1 <= args.merge_packet_workers <= MAX_MERGE_PACKET_WORKERS:
        print(
            json.dumps(
                {
                    "action": "error",
                    "error": f"--merge-packet-workers must be between 1 and {MAX_MERGE_PACKET_WORKERS}",
                }
            ),
            file=sys.stderr,
        )
        return EXIT_FAILURE

    return run_manifest(
        list_prs=lambda: default_list_prs(args.repo, args.limit),
        merge_packet=(lambda pr: default_merge_packet(pr, args.repo))
        if args.with_merge_packets
        else None,
        merge_packet_workers=args.merge_packet_workers,
        inventory_candidates=default_inventory_candidates if args.include_worktrees else None,
        limit=args.limit,
        stale_days=args.stale_days,
        out_file=args.out,
        summary=args.summary,
    )


if __name__ == "__main__":
    sys.exit(main())

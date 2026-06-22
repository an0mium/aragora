#!/usr/bin/env python3
"""Stale-tail classifier + restack queue for codex/ PRs (NON-DESTRUCTIVE).

The autonomous PR pipeline accumulates a tail of old open ``codex/`` PRs that
nothing manages: some merely need a restack (merge conflicts), some are
blocked on red CI, some are promotable and just need
``scripts/pr_ready_triage.py`` to pick them up. This janitor classifies that
tail and queues restack work — it NEVER closes PRs, NEVER deletes branches,
NEVER pushes. The only GitHub mutation it can perform (gated behind
``--apply``) is posting one idempotent explanatory comment per PR.

Selection: open PRs (draft or ready) whose head branch matches
``--branch-prefix`` (default ``codex/``), created at least ``--stale-days``
ago (default 4).

Classification (per selected PR):

- ``active``      — updated within ``--inactive-days`` (default 2): skip.
- ``promotable``  — MERGEABLE with green (non-failing, non-pending) checks:
  emit a recommendation to run ``scripts/pr_ready_triage.py``; no queue entry,
  no comment.
- ``restack``     — ``mergeable == CONFLICTING``: queue entry (+ comment with
  ``--apply``).
- ``blocked``     — failing checks: queue entry naming the failing checks
  (+ comment with ``--apply``).
- ``indeterminate`` — pending checks / unknown mergeability: skip (fail safe).

Restack queue: ``.aragora/restack-queue.json`` holds a list of
``{pr, head_ref, classification, reason, detected_at}``. The janitor merges
with the existing file, dedupes by PR number (this run's classification wins
for PRs it classified), and never drops entries it did not classify this run.
A corrupt existing queue file is an error (exit 1) — it is never overwritten.
The queue file is the janitor's local work product and is written in both
modes; ``--apply`` gates only the GitHub comment mutation.

Idempotent comments: each janitor comment embeds the marker
``<!-- stale-pr-janitor -->``. If any existing comment on the PR already
contains the marker, the janitor skips posting (checked via
``gh pr view N --json comments``, read-only). ``--max-comments`` (default 10)
caps posts per run.

Safety model (mirrors ``boss_pr_janitor.py`` / ``auto_evidence_cycle.py``):
dry-run by default printing a JSON-lines plan; ``--apply`` gates mutations;
3 consecutive identical comment failures trip the breaker (exit 2); any
failed mutation fails closed (exit 1); clean runs exit 0.

Stdlib-only by design so it can run anywhere ``gh`` is authenticated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

DEFAULT_REPO = "synaptent/aragora"
DEFAULT_BRANCH_PREFIX = "codex/"
DEFAULT_STALE_DAYS = 4
DEFAULT_INACTIVE_DAYS = 2
DEFAULT_MAX_COMMENTS = 10
DEFAULT_BREAKER_THRESHOLD = 3
DEFAULT_QUEUE_FILE = os.path.join(".aragora", "restack-queue.json")
GH_TIMEOUT_SECONDS = 120

COMMENT_MARKER = "<!-- stale-pr-janitor -->"

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_BREAKER = 2

CLASS_ACTIVE = "active"
CLASS_PROMOTABLE = "promotable"
CLASS_RESTACK = "restack"
CLASS_BLOCKED = "blocked"
CLASS_INDETERMINATE = "indeterminate"

# Classifications that produce a queue entry (and, with --apply, a comment).
QUEUED_CLASSIFICATIONS = frozenset({CLASS_RESTACK, CLASS_BLOCKED})

# Check states/conclusions treated as failing (same set as boss_pr_janitor).
_FAILING_STATES = frozenset(
    {
        "FAILURE",
        "ERROR",
        "CANCELLED",
        "TIMED_OUT",
        "ACTION_REQUIRED",
        "STARTUP_FAILURE",
    }
)

_PENDING_STATUSES = frozenset({"IN_PROGRESS", "QUEUED", "PENDING", "WAITING", "REQUESTED"})

_GH_LIST_FIELDS = "number,headRefName,isDraft,title,mergeable,createdAt,updatedAt,statusCheckRollup"


# --- Classification ---------------------------------------------------------------


def _parse_iso(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def failing_check_names(pr: dict[str, Any]) -> list[str]:
    rollup = pr.get("statusCheckRollup") or []
    if not isinstance(rollup, list):
        return []
    names: list[str] = []
    for check in rollup:
        if not isinstance(check, dict):
            continue
        state = str(check.get("state") or "").upper()
        conclusion = str(check.get("conclusion") or "").upper()
        if state in _FAILING_STATES or conclusion in _FAILING_STATES:
            names.append(str(check.get("name") or check.get("context") or "(unnamed check)"))
    return names


def _has_pending_checks(pr: dict[str, Any]) -> bool:
    rollup = pr.get("statusCheckRollup") or []
    if not isinstance(rollup, list):
        return False
    for check in rollup:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "").upper()
        state = str(check.get("state") or "").upper()
        if status in _PENDING_STATUSES or state in _PENDING_STATUSES:
            return True
    return False


def is_stale(pr: dict[str, Any], *, stale_days: int, now: datetime) -> bool:
    """Fail safe: an unparseable/missing createdAt is never selected."""
    created = _parse_iso(pr.get("createdAt"))
    if created is None:
        return False
    return now - created >= timedelta(days=max(0, stale_days))


def classify(pr: dict[str, Any], *, inactive_days: int, now: datetime) -> dict[str, Any]:
    """Classify one stale PR. Pure function; never touches the network."""
    number = int(pr.get("number") or 0)
    head_ref = str(pr.get("headRefName") or "")
    updated = _parse_iso(pr.get("updatedAt"))
    # Fail safe: unparseable updatedAt counts as recent activity (skip).
    if updated is None or now - updated < timedelta(days=max(0, inactive_days)):
        return {
            "pr": number,
            "head_ref": head_ref,
            "classification": CLASS_ACTIVE,
            "reason": f"updated within {inactive_days}d (recent human/agent activity)",
        }
    mergeable = str(pr.get("mergeable") or "").upper()
    if mergeable == "CONFLICTING":
        return {
            "pr": number,
            "head_ref": head_ref,
            "classification": CLASS_RESTACK,
            "reason": "mergeable=CONFLICTING; needs restack onto main",
        }
    failing = failing_check_names(pr)
    if failing:
        return {
            "pr": number,
            "head_ref": head_ref,
            "classification": CLASS_BLOCKED,
            "reason": f"failing checks: {', '.join(sorted(failing)[:5])}",
            "failing_checks": sorted(failing),
        }
    if mergeable == "MERGEABLE" and not _has_pending_checks(pr):
        return {
            "pr": number,
            "head_ref": head_ref,
            "classification": CLASS_PROMOTABLE,
            "reason": ("mergeable with green checks; run scripts/pr_ready_triage.py to promote"),
        }
    return {
        "pr": number,
        "head_ref": head_ref,
        "classification": CLASS_INDETERMINATE,
        "reason": f"checks pending or mergeable={mergeable or '(unknown)'}; skipping (fail safe)",
    }


# --- Restack queue ------------------------------------------------------------------


def load_queue(path: str) -> list[dict[str, Any]]:
    """Load the existing queue. A corrupt file raises (never overwritten)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    if not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"corrupt restack queue {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"corrupt restack queue {path}: expected a JSON list")
    return payload


def merge_queue(
    existing: list[dict[str, Any]], new_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge new entries into the queue, deduped by PR number.

    Entries this run classified replace any prior entry for the same PR;
    entries it did not classify are never dropped. Unrecognizable existing
    entries (no integer ``pr``) are preserved verbatim, first.
    """
    preserved_opaque: list[dict[str, Any]] = []
    by_pr: dict[int, dict[str, Any]] = {}
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        try:
            number = int(entry["pr"])
        except (KeyError, TypeError, ValueError):
            preserved_opaque.append(entry)
            continue
        by_pr[number] = entry
    for entry in new_entries:
        by_pr[int(entry["pr"])] = entry
    return preserved_opaque + [by_pr[number] for number in sorted(by_pr)]


def write_queue(path: str, entries: list[dict[str, Any]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, sort_keys=True)
        fh.write("\n")


# --- Comments -------------------------------------------------------------------------


def comment_body(entry: dict[str, Any]) -> str:
    return (
        f"{COMMENT_MARKER}\n"
        f"stale-pr-janitor classified this PR as **{entry['classification']}** "
        f"({entry['reason']}).\n\n"
        "It has been queued in `.aragora/restack-queue.json` for a restack pass. "
        "This janitor is non-destructive: it never closes PRs, never deletes "
        "branches, and never pushes. Rebase or update the branch to clear the "
        "classification."
    )


def has_janitor_comment(comments: list[str]) -> bool:
    return any(COMMENT_MARKER in body for body in comments)


# --- Default (real) I/O callables -----------------------------------------------------


def default_list_prs(repo: str) -> list[dict[str, Any]]:
    """One ``gh pr list`` call covering drafts and ready PRs (read-only)."""
    command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--json",
        _GH_LIST_FIELDS,
        "--limit",
        "200",
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


def default_fetch_comments(repo: str, pr: int) -> list[str]:
    """Existing comment bodies via ``gh pr view`` (read-only). Raises on failure
    so the caller fails closed (no post when idempotency cannot be verified)."""
    command = ["gh", "pr", "view", str(pr), "--repo", repo, "--json", "comments"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=GH_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr view {pr} failed (exit {result.returncode})")
    payload = json.loads(result.stdout or "{}")
    comments = payload.get("comments") if isinstance(payload, dict) else None
    if not isinstance(comments, list):
        return []
    return [str(c.get("body") or "") for c in comments if isinstance(c, dict)]


def default_post_comment(repo: str, pr: int, body: str) -> tuple[bool, str]:
    """The only GitHub mutation: ``gh pr comment``. Returns (ok, error)."""
    command = ["gh", "pr", "comment", str(pr), "--repo", repo, "--body", body]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}"
    if result.returncode != 0:
        return False, result.stderr.strip()[:300] or f"gh pr comment exited {result.returncode}"
    return True, ""


# --- Orchestrator ----------------------------------------------------------------------


def run_janitor(
    *,
    list_prs: Callable[[], list[dict[str, Any]]],
    fetch_comments: Callable[[int], list[str]],
    post_comment: Callable[[int, str], tuple[bool, str]],
    apply: bool,
    queue_file: str = DEFAULT_QUEUE_FILE,
    branch_prefix: str = DEFAULT_BRANCH_PREFIX,
    stale_days: int = DEFAULT_STALE_DAYS,
    inactive_days: int = DEFAULT_INACTIVE_DAYS,
    max_comments: int = DEFAULT_MAX_COMMENTS,
    breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Classify the stale tail, update the restack queue, optionally comment.

    Non-destructive by construction: the only actions this function can emit
    are queue-file writes, recommendations, and (with ``apply``) idempotent
    PR comments. There is no code path that closes, deletes, or pushes.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "classified": [],
        "queued": [],
        "recommendations": [],
        "comments_posted": [],
        "comments_skipped_existing": [],
        "failed": [],
        "queue_file": queue_file,
        "breaker_tripped": False,
        "exit_code": EXIT_OK,
    }

    selected = [
        pr
        for pr in list_prs()
        if isinstance(pr, dict)
        and str(pr.get("headRefName") or "").startswith(branch_prefix)
        and is_stale(pr, stale_days=stale_days, now=now)
    ]
    selected.sort(key=lambda pr: int(pr.get("number") or 0))

    detected_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    queue_entries: list[dict[str, Any]] = []
    for pr in selected:
        entry = classify(pr, inactive_days=inactive_days, now=now)
        summary["classified"].append(entry)
        log(json.dumps({"action": "classify", **entry, "dry_run": not apply}))
        if entry["classification"] == CLASS_PROMOTABLE:
            summary["recommendations"].append(
                {
                    "pr": entry["pr"],
                    "recommendation": "run scripts/pr_ready_triage.py (no action taken here)",
                }
            )
        if entry["classification"] in QUEUED_CLASSIFICATIONS:
            queue_entries.append(
                {
                    "pr": entry["pr"],
                    "head_ref": entry["head_ref"],
                    "classification": entry["classification"],
                    "reason": entry["reason"],
                    "detected_at": detected_at,
                }
            )

    for rec in summary["recommendations"]:
        log(json.dumps({"action": "recommend_promote", **rec}))

    if queue_entries:
        try:
            existing = load_queue(queue_file)
            merged = merge_queue(existing, queue_entries)
            write_queue(queue_file, merged)
        except (RuntimeError, OSError) as exc:
            summary["exit_code"] = EXIT_FAILURES
            log(json.dumps({"action": "queue_error", "error": str(exc)[:300]}))
            _log_summary(summary, log)
            return summary
        summary["queued"] = [entry["pr"] for entry in queue_entries]
        log(json.dumps({"action": "queue_written", "path": queue_file, "entries": len(merged)}))

    identical_errors = 0
    last_error: str | None = None
    comment_candidates = queue_entries[: max(0, max_comments)]
    for entry in comment_candidates:
        number = entry["pr"]
        try:
            existing_comments = fetch_comments(number)
        except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
            # Fail closed: never post when idempotency cannot be verified.
            summary["failed"].append(number)
            error = f"comment check failed: {str(exc)[:200]}"
            log(json.dumps({"action": "comment_failed", "pr": number, "error": error}))
            identical_errors = identical_errors + 1 if error == last_error else 1
            last_error = error
            if identical_errors >= breaker_threshold:
                summary["breaker_tripped"] = True
                break
            continue
        if has_janitor_comment(existing_comments):
            summary["comments_skipped_existing"].append(number)
            log(json.dumps({"action": "comment_skipped", "pr": number, "reason": "marker exists"}))
            continue
        if not apply:
            log(json.dumps({"action": "comment", "pr": number, "dry_run": True}))
            continue
        ok, error = post_comment(number, comment_body(entry))
        if ok:
            summary["comments_posted"].append(number)
            identical_errors = 0
            last_error = None
            log(json.dumps({"action": "comment", "pr": number, "dry_run": False}))
            continue
        summary["failed"].append(number)
        log(json.dumps({"action": "comment_failed", "pr": number, "error": error[:300]}))
        identical_errors = identical_errors + 1 if error == last_error else 1
        last_error = error
        if identical_errors >= breaker_threshold:
            summary["breaker_tripped"] = True
            log(
                json.dumps(
                    {
                        "action": "breaker_tripped",
                        "identical_errors": identical_errors,
                        "error": error[:300],
                    }
                )
            )
            break

    if summary["breaker_tripped"]:
        summary["exit_code"] = EXIT_BREAKER
    elif summary["failed"]:
        summary["exit_code"] = EXIT_FAILURES

    _log_summary(summary, log)
    return summary


def _log_summary(summary: dict[str, Any], log: Callable[[str], None]) -> None:
    log(
        json.dumps(
            {
                "action": "summary",
                "mode": summary["mode"],
                "classified": len(summary["classified"]),
                "queued": summary["queued"],
                "recommendations": [rec["pr"] for rec in summary["recommendations"]],
                "comments_posted": summary["comments_posted"],
                "comments_skipped_existing": summary["comments_skipped_existing"],
                "failed": summary["failed"],
                "breaker_tripped": summary["breaker_tripped"],
                "non_destructive": "never closes PRs, never deletes branches, never pushes",
            }
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stale-tail classifier + restack queue for codex/ PRs. "
            "Non-destructive; dry-run by default; --apply gates the only "
            "mutation (one idempotent PR comment per classified PR)."
        )
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo (owner/name)")
    parser.add_argument(
        "--branch-prefix",
        default=DEFAULT_BRANCH_PREFIX,
        help=f"Head-branch prefix to consider (default {DEFAULT_BRANCH_PREFIX!r})",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=DEFAULT_STALE_DAYS,
        help=f"Only consider PRs at least this many days old (default {DEFAULT_STALE_DAYS})",
    )
    parser.add_argument(
        "--inactive-days",
        type=int,
        default=DEFAULT_INACTIVE_DAYS,
        help=f"PRs updated within this many days are 'active' and skipped "
        f"(default {DEFAULT_INACTIVE_DAYS})",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=DEFAULT_MAX_COMMENTS,
        help=f"Maximum PR comments per run (default {DEFAULT_MAX_COMMENTS})",
    )
    parser.add_argument(
        "--queue-file",
        default=DEFAULT_QUEUE_FILE,
        help=f"Restack queue path (default {DEFAULT_QUEUE_FILE})",
    )
    parser.add_argument(
        "--breaker-threshold",
        type=int,
        default=DEFAULT_BREAKER_THRESHOLD,
        help=f"Consecutive identical comment failures that abort the run "
        f"(default {DEFAULT_BREAKER_THRESHOLD})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually post idempotent PR comments (default: dry-run; the local "
        "restack queue file is written in both modes)",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_janitor(
            list_prs=lambda: default_list_prs(args.repo),
            fetch_comments=lambda pr: default_fetch_comments(args.repo, pr),
            post_comment=lambda pr, body: default_post_comment(args.repo, pr, body),
            apply=args.apply,
            queue_file=args.queue_file,
            branch_prefix=args.branch_prefix,
            stale_days=args.stale_days,
            inactive_days=args.inactive_days,
            max_comments=args.max_comments,
            breaker_threshold=max(1, args.breaker_threshold),
        )
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"action": "error", "error": str(exc)[:500]}), file=sys.stderr)
        return EXIT_FAILURES
    return int(summary["exit_code"])


if __name__ == "__main__":
    sys.exit(main())

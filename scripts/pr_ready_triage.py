#!/usr/bin/env python3
"""Bounded draft→ready promoter for writer-agent PRs (pipeline gap fix).

Writer agents produce ``codex/`` branches and the publisher opens DRAFT PRs,
but nothing in the pipeline promotes those drafts to ready — so the downstream
machinery (``scripts/auto_evidence_cycle.py`` evidence minting,
``scripts/quorum_rerun_reconciler.py`` quorum reruns,
``scripts/auto_merge_bucket_a.py`` merges) starves. This script closes that
gap autonomously and *boundedly*: it selects qualifying draft PRs and runs
``gh pr ready`` on at most ``--max-promotions`` of them per invocation.

Probe (cheapest reliable, two stages — mirrors ``auto_evidence_cycle.py``):

1. One ``gh pr list --draft`` call prunes obviously unpromotable drafts:
   wrong branch prefix, failing/errored checks, not MERGEABLE, blocking
   labels (``merge:manual``, ``do-not-promote``, ``held``), draft-by-design
   governance markers in title/body (hard never-promote, see below), and PRs
   younger than ``--min-age-minutes`` (lets draft CI finish).
2. Each survivor (up to ``--max-scan``) is confirmed against the gate's own
   ground truth: ``review-queue merge-packet --pr N --json``. Only Tier 0-2
   entries with no ``requires_human_risk_settlement`` and
   ``unresolved_dissent`` false qualify. A missing/unknown tier is a REJECT
   (fail closed) — the same rule ``auto_evidence_cycle.needs_evidence``
   applies.

Hard never-promote: governance PRs that are draft by design — Tier 4
settlements, operator-settlement-required changes, anything mentioning the
Scarmani escalation track — must NEVER be marked ready by automation. Any PR
whose title or body matches (case-insensitive)
``tier\\s*4 | draft by design | do not mark ready | operator settlement
required | scarmani`` is excluded in stage 1 before any packet probe.

Safety model (mirrors ``boss_pr_janitor.py`` / ``auto_evidence_cycle.py``):

- Dry-run by default: the plan is printed as JSON lines, nothing mutates.
  ``--apply`` gates ALL mutations (the only mutation is ``gh pr ready``).
- Per-invocation caps: ``--max-promotions`` (default 5), ``--max-scan``
  packet probes (default 15).
- Identical-error breaker: 3 consecutive identical promotion failures abort
  the run (exit 2) — a systemic fault (gh auth broken) must not burn the cap.
- Fail-closed exits: 0 clean (including empty plan), 1 any failure,
  2 breaker tripped.

Downstream: promoted PRs are picked up by ``scripts/auto_evidence_cycle.py``
on the next arbiter pass; this script takes no further action after ``gh pr
ready``.

Stdlib-only by design so it can run anywhere ``gh`` is authenticated.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from itertools import zip_longest
from typing import Any, Callable

DEFAULT_REPO = "synaptent/aragora"
# ``benchmark-truth-publication/`` is the daily proof-surface publisher
# (.github/workflows/benchmark-truth-publication.yml), which always opens its PR
# as a draft. Promoting it has been a manual step nobody owns, and while it
# stays a draft every merge path skips it (they all filter drafts), so the
# recurring surface goes stale whenever no one is watching — the drift issue
# #9519 was filed for. Stage-2's tier and dissent gates still decide; this only
# makes them reachable. The namespace carries no push-side provenance
# (check_branch_mutation_policy.py lints workflow YAML; it does not restrict who
# may push a matching branch), so the governance markers, blocking labels and
# the merge-packet tier check remain the actual defenses.
DEFAULT_BRANCH_PREFIXES = ("codex/", "benchmark-truth-publication/")
DEFAULT_MIN_AGE_MINUTES = 30
DEFAULT_MAX_SCAN = 15
DEFAULT_MAX_PROMOTIONS = 5
DEFAULT_BREAKER_THRESHOLD = 3
GH_TIMEOUT_SECONDS = 120
PACKET_TIMEOUT_SECONDS = 300

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_BREAKER = 2

# Merge-packet tiers eligible for autonomous promotion. Same band as
# auto_evidence_cycle.AUTO_POSTABLE_TIERS; missing/unknown tier always rejects.
PROMOTABLE_TIERS = frozenset({0, 1, 2})

# Labels that veto promotion outright (stage 1).
BLOCKING_LABELS = frozenset({"merge:manual", "do-not-promote", "held"})

# Draft-by-design governance markers (title or body, case-insensitive).
# These PRs are draft on purpose — Tier 4 settlements, operator-settlement
# flows, the Scarmani escalation track — and must NEVER be promoted.
NEVER_PROMOTE_RE = re.compile(
    r"tier\s*4|draft by design|do not mark ready|operator settlement required|scarmani",
    re.IGNORECASE,
)

# Zero-width characters stripped before the never-promote match so markers
# can't be split invisibly (ZWSP, ZWNJ, ZWJ, BOM/ZWNBSP, word joiner).
# All Unicode format chars (category Cf: zero-widths, soft hyphen, BOM, ...)
# are stripped, and every separator (category Zs, incl. the one space NFKC
# does not fold, U+1680 OGHAM SPACE MARK) maps to a plain ASCII space.
_FORMAT_CHARS_RE = re.compile(
    r"[\u00ad\u0600-\u0605\u061c\u06dd\u070f\u08e2\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff\ufff9-\ufffb]"
)
_SPACE_SEPARATORS_RE = re.compile(r"[\u1680\u2000-\u200a\u202f\u205f\u3000\u00a0]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

# Check states/conclusions that disqualify a draft (same set as
# boss_pr_janitor._FAILING_STATES).
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

_GH_LIST_FIELDS = (
    "number,headRefName,title,body,labels,statusCheckRollup,mergeable,createdAt,updatedAt"
)


# --- Stage 1: cheap pruning ------------------------------------------------------


def _labels(pr: dict[str, Any]) -> set[str]:
    raw = pr.get("labels") or []
    out: set[str] = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        name = str(item.get("name") or "") if isinstance(item, dict) else str(item)
        if name:
            out.add(name)
    return out


def has_failing_checks(pr: dict[str, Any]) -> bool:
    rollup = pr.get("statusCheckRollup") or []
    if not isinstance(rollup, list):
        return False
    for check in rollup:
        if not isinstance(check, dict):
            continue
        state = str(check.get("state") or "").upper()
        conclusion = str(check.get("conclusion") or "").upper()
        if state in _FAILING_STATES or conclusion in _FAILING_STATES:
            return True
    return False


def _normalize_marker_text(text: str) -> str:
    """Normalize against Unicode evasion before the never-promote match.

    NFKC folds fullwidth/compatibility forms (``Ｔｉｅｒ ４`` -> ``Tier 4``)
    and most exotic spaces; format characters (zero-widths, soft hyphen) are
    stripped so markers can't be split invisibly; remaining separators map to
    ASCII space and runs collapse so literal-space markers still match; and
    ``casefold`` handles aggressive case mappings beyond IGNORECASE.

    This guard is best-effort by nature (an intra-word split like
    ``settle ment`` still evades it); the merge-packet tier check in
    ``packet_blocker`` is the sole authority — this layer only adds friction.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _FORMAT_CHARS_RE.sub("", text)
    text = _SPACE_SEPARATORS_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).casefold()


def is_draft_by_design(pr: dict[str, Any]) -> bool:
    """Hard never-promote: governance markers in title or body."""
    text = _normalize_marker_text(f"{pr.get('title') or ''}\n{pr.get('body') or ''}")
    return NEVER_PROMOTE_RE.search(text) is not None


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


def _too_young(pr: dict[str, Any], *, min_age_minutes: int, now: datetime) -> bool:
    """Fail closed: an unparseable/missing createdAt counts as too young."""
    created = _parse_iso(pr.get("createdAt"))
    if created is None:
        return True
    return now - created < timedelta(minutes=max(0, min_age_minutes))


def stage1_blocker(
    pr: dict[str, Any],
    *,
    branch_prefixes: tuple[str, ...],
    min_age_minutes: int,
    now: datetime,
) -> str | None:
    """Return the stage-1 prune reason, or None if the draft survives."""
    head = str(pr.get("headRefName") or "")
    if not any(head.startswith(prefix) for prefix in branch_prefixes):
        return f"branch prefix not in {list(branch_prefixes)}"
    if is_draft_by_design(pr):
        return "draft-by-design governance marker (hard never-promote)"
    blocked = sorted(_labels(pr) & BLOCKING_LABELS)
    if blocked:
        return f"blocking label ({blocked[0]})"
    if has_failing_checks(pr):
        return "failing/errored check in rollup"
    mergeable = str(pr.get("mergeable") or "")
    if mergeable != "MERGEABLE":
        return f"mergeable={mergeable or '(unknown)'} (requires MERGEABLE)"
    if _too_young(pr, min_age_minutes=min_age_minutes, now=now):
        return f"younger than {min_age_minutes}m (draft CI may still be running)"
    return None


def stage1_candidates(
    prs: list[dict[str, Any]],
    *,
    branch_prefixes: tuple[str, ...],
    min_age_minutes: int,
    now: datetime,
) -> list[dict[str, Any]]:
    """Drafts surviving the cheap prune: oldest first, round-robin by prefix.

    Oldest-first *within* a prefix, but interleaved *between* prefixes. A
    single global PR-number sort lets one busy namespace consume the whole
    ``--max-scan`` budget and starve another indefinitely: with 16 open
    ``codex/`` drafts and a scan budget of 15, the daily publication PR — always
    the highest number — would never be probed at all.
    """
    buckets: dict[str, list[dict[str, Any]]] = {prefix: [] for prefix in branch_prefixes}
    for pr in prs:
        if not isinstance(pr, dict):
            continue
        try:
            int(pr["number"])
        except (KeyError, TypeError, ValueError):
            continue
        blocker = stage1_blocker(
            pr, branch_prefixes=branch_prefixes, min_age_minutes=min_age_minutes, now=now
        )
        if blocker is not None:
            continue
        head = str(pr.get("headRefName") or "")
        for prefix in branch_prefixes:
            if head.startswith(prefix):
                buckets[prefix].append(pr)
                break

    for bucket in buckets.values():
        bucket.sort(key=lambda pr: int(pr["number"]))

    interleaved: list[dict[str, Any]] = []
    for row in zip_longest(*buckets.values()):
        interleaved.extend(pr for pr in row if pr is not None)
    return interleaved


# --- Stage 2: merge-packet ground truth -------------------------------------------
#
# Field names come from the canonical merge-packet entries that
# auto_evidence_cycle.needs_evidence reads: ``tier``,
# ``requires_human_risk_settlement``, ``unresolved_dissent`` (and ``pr_number``
# for entry matching in default_fetch_packet). Do not invent new keys.


def packet_blocker(entry: dict[str, Any]) -> str | None:
    """Return the stage-2 reject reason, or None when promotion is allowed.

    Fail closed: an empty packet or a missing/unparseable tier always rejects.
    """
    if not entry:
        return "empty merge packet (fail closed)"
    # AUTHORITATIVE gate: the merge-packet tier decides promotability. The
    # stage-1 NEVER_PROMOTE_RE text match is advisory defense-in-depth only —
    # do not weaken this check on the assumption that the regex backstops it.
    raw_tier = entry.get("tier")
    if not isinstance(raw_tier, (int, str)):
        return f"unknown tier {raw_tier!r} (fail closed)"
    try:
        tier = int(raw_tier)
    except (TypeError, ValueError):
        return f"unknown tier {raw_tier!r} (fail closed)"
    if tier not in PROMOTABLE_TIERS:
        return f"tier {tier} outside promotable band {sorted(PROMOTABLE_TIERS)}"
    if entry.get("requires_human_risk_settlement"):
        return "requires_human_risk_settlement"
    if entry.get("unresolved_dissent"):
        return "unresolved_dissent"
    return None


# --- Default (real) I/O callables --------------------------------------------------


def default_list_draft_prs(repo: str) -> list[dict[str, Any]]:
    """One cheap ``gh pr list --draft`` call (read-only)."""
    command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--draft",
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


def default_fetch_packet(repo: str, pr: int) -> dict[str, Any]:
    """Canonical per-PR probe: ``review-queue merge-packet --pr N --json``.

    Same entry-unwrapping logic as ``auto_evidence_cycle.default_fetch_packet``.
    Returns ``{}`` on any failure (which stage 2 rejects, fail closed).
    """
    try:
        proc = subprocess.run(
            [
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
            ],
            capture_output=True,
            text=True,
            timeout=PACKET_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        entries = payload.get("entries")
        entries = entries if isinstance(entries, list) else [payload]
    elif isinstance(payload, list):
        entries = payload
    else:
        return {}
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("pr_number")) == str(pr):
            return entry
    return {}


def default_mark_ready(repo: str, pr: int) -> tuple[bool, str]:
    """The only mutation: ``gh pr ready N``. Returns (ok, error)."""
    command = ["gh", "pr", "ready", str(pr), "--repo", repo]
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
        return False, result.stderr.strip()[:300] or f"gh pr ready exited {result.returncode}"
    return True, ""


# --- Orchestrator -------------------------------------------------------------------


def run_triage(
    *,
    list_prs: Callable[[], list[dict[str, Any]]],
    fetch_packet: Callable[[int], dict[str, Any]],
    mark_ready: Callable[[int], tuple[bool, str]],
    apply: bool,
    branch_prefixes: tuple[str, ...] = DEFAULT_BRANCH_PREFIXES,
    min_age_minutes: int = DEFAULT_MIN_AGE_MINUTES,
    max_scan: int = DEFAULT_MAX_SCAN,
    max_promotions: int = DEFAULT_MAX_PROMOTIONS,
    breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Plan and (with ``apply``) execute one bounded promotion pass."""
    if now is None:
        now = datetime.now(timezone.utc)

    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "plan": [],
        "rejected": [],
        "promoted": [],
        "failed": [],
        "breaker_tripped": False,
        "exit_code": EXIT_OK,
    }

    survivors = stage1_candidates(
        list_prs(),
        branch_prefixes=branch_prefixes,
        min_age_minutes=min_age_minutes,
        now=now,
    )

    probes = 0
    for pr in survivors:
        if len(summary["plan"]) >= max_promotions or probes >= max_scan:
            break
        probes += 1
        number = int(pr["number"])
        entry = fetch_packet(number)
        blocker = packet_blocker(entry)
        if blocker is not None:
            summary["rejected"].append({"pr": number, "reason": blocker})
            log(json.dumps({"action": "reject", "pr": number, "reason": blocker}))
            continue
        item = {
            "action": "promote",
            "pr": number,
            "head_ref": pr.get("headRefName"),
            "title": str(pr.get("title") or "")[:80],
            "tier": entry.get("tier"),
        }
        summary["plan"].append(item)
        log(json.dumps({**item, "dry_run": not apply}))

    if apply:
        identical_errors = 0
        last_error: str | None = None
        for item in summary["plan"]:
            ok, error = mark_ready(item["pr"])
            if ok:
                summary["promoted"].append(item["pr"])
                identical_errors = 0
                last_error = None
                log(json.dumps({"action": "promoted", "pr": item["pr"]}))
                continue
            summary["failed"].append(item["pr"])
            log(json.dumps({"action": "promote_failed", "pr": item["pr"], "error": error[:300]}))
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

    log(
        json.dumps(
            {
                "action": "summary",
                "mode": summary["mode"],
                "stage1_survivors": len(survivors),
                "packet_probes": probes,
                "planned_promotions": len(summary["plan"]),
                "rejected": len(summary["rejected"]),
                "promoted": summary["promoted"],
                "failed": summary["failed"],
                "breaker_tripped": summary["breaker_tripped"],
                "downstream": (
                    "promoted PRs are picked up by scripts/auto_evidence_cycle.py "
                    "on the next arbiter pass; no further action here"
                ),
            }
        )
    )
    return summary


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def repo_arg(value: str) -> str:
    """Argparse type: validate ``owner/name`` at parse time, before any gh call."""
    if not _REPO_RE.match(value):
        raise argparse.ArgumentTypeError(f"--repo must look like owner/name, got {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded draft-to-ready promoter for writer-agent PRs. Dry-run by "
            "default; --apply gates the only mutation (gh pr ready)."
        )
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, type=repo_arg, help="GitHub repo (owner/name)"
    )
    parser.add_argument(
        "--branch-prefix",
        action="append",
        default=None,
        help=f"Head-branch prefix to consider; repeatable (default: {list(DEFAULT_BRANCH_PREFIXES)})",
    )
    parser.add_argument(
        "--min-age-minutes",
        type=int,
        default=DEFAULT_MIN_AGE_MINUTES,
        help=f"Skip drafts younger than this, letting draft CI finish (default {DEFAULT_MIN_AGE_MINUTES})",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=DEFAULT_MAX_SCAN,
        help=f"Maximum merge-packet probes per run (default {DEFAULT_MAX_SCAN})",
    )
    parser.add_argument(
        "--max-promotions",
        type=int,
        default=DEFAULT_MAX_PROMOTIONS,
        help=f"Maximum PRs promoted per run (default {DEFAULT_MAX_PROMOTIONS})",
    )
    parser.add_argument(
        "--breaker-threshold",
        type=int,
        default=DEFAULT_BREAKER_THRESHOLD,
        help=f"Consecutive identical promotion failures that abort the run (default {DEFAULT_BREAKER_THRESHOLD})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually run gh pr ready (default: dry-run, read-only)",
    )
    args = parser.parse_args(argv)

    prefixes = tuple(args.branch_prefix) if args.branch_prefix else DEFAULT_BRANCH_PREFIXES

    try:
        summary = run_triage(
            list_prs=lambda: default_list_draft_prs(args.repo),
            fetch_packet=lambda pr: default_fetch_packet(args.repo, pr),
            mark_ready=lambda pr: default_mark_ready(args.repo, pr),
            apply=args.apply,
            branch_prefixes=prefixes,
            min_age_minutes=args.min_age_minutes,
            max_scan=max(0, args.max_scan),
            max_promotions=max(0, args.max_promotions),
            breaker_threshold=max(1, args.breaker_threshold),
        )
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"action": "error", "error": str(exc)[:500]}), file=sys.stderr)
        return EXIT_FAILURES
    return int(summary["exit_code"])


if __name__ == "__main__":
    sys.exit(main())

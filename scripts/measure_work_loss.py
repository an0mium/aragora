#!/usr/bin/env python3
"""Measure agent work-loss (the waste ratio) — the leverage ratio's dual.

Sprint 4 goal 2 / Steering Leverage Operating Plan v2, Pillar 7 MVP.

Waste accounting from three injectable inputs:
1. Outbox dir items (live + archive, with publication state),
2. ``git ls-remote --heads origin`` output (or a captured file),
3. A PR list with head refs / state / merge timestamps (or a captured file).

Computed units (defined in ``UNIT_DEFINITIONS`` and printed in --json output):
``branches_pushed_never_prd``, ``outbox_expired_unpublished``,
``outbox_lost_never_pushed``, ``prs_closed_unmerged``, ``produced_units``.

``waste_ratio = lost_units / max(1, produced_units)``; loss categories are
de-duplicated by unit key (branch name when present, else outbox idempotency
key) so a unit of lost work counts in exactly one category.

``--publish`` updates the waste section of ``docs/status/LEVERAGE.md`` via the
same managed-region renderer as measure_leverage_ratio.py; manual text outside
the managed region is never touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from measure_leverage_ratio import (  # type: ignore[no-redef]
        DEFAULT_REPO,
        DEFAULT_STATUS_DOC,
        update_leverage_md,
    )
else:  # imported as scripts.measure_work_loss
    from scripts.measure_leverage_ratio import (
        DEFAULT_REPO,
        DEFAULT_STATUS_DOC,
        update_leverage_md,
    )

METHODOLOGY_VERSION = 1
DEFAULT_OUTBOX_DIRS = (
    ".aragora/automation-outbox",
    ".aragora/automation-outbox/archive",
    ".aragora/automation-outbox-archive",
)
PROTECTED_BRANCHES = {"main", "master", "gh-pages", "HEAD"}

UNIT_DEFINITIONS = {
    "branches_pushed_never_prd": (
        "Non-protected branch present on origin that has never been the head "
        "ref of any PR and is not already claimed by an outbox loss category."
    ),
    "outbox_expired_unpublished": (
        "Outbox item (live or archive) whose expires_at has passed and which "
        "never reached a published state (no explicit publication marker, no "
        "PR for its branch, not marked already-satisfied)."
    ),
    "outbox_lost_never_pushed": (
        "Unpublished outbox item whose branch never reached origin — the work "
        "exists (or existed) only locally."
    ),
    "prs_closed_unmerged": ("PR closed without merge, with closed_at inside the window."),
    "produced_units": "PRs merged inside the window (merged_at >= window start).",
    "lost_units": (
        "Sum of the four loss categories after de-duplication by unit key "
        "(branch name when present, else outbox idempotency key); each lost "
        "unit counts in exactly one category."
    ),
    "waste_ratio": "lost_units / max(1, produced_units).",
}


# ---------------------------------------------------------------------------
# Input loading / subprocess boundaries (all injectable via CLI files)
# ---------------------------------------------------------------------------


def run_ls_remote() -> str:
    proc = subprocess.run(
        ["git", "ls-remote", "--heads", "origin"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return proc.stdout


def fetch_all_prs(repo: str) -> list[dict]:
    """Fetch head-ref/state/merge info for all PRs via gh REST (paginated)."""
    proc = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/pulls?state=all&per_page=100",
            "--jq",
            ".[] | {number: .number, head_ref: .head.ref, state: .state, "
            "merged: (.merged_at != null), merged_at: .merged_at, "
            "closed_at: .closed_at}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=1800,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def parse_ls_remote(text: str) -> set[str]:
    """Parse ``git ls-remote --heads`` output into a set of branch names."""
    heads: set[str] = set()
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            heads.add(parts[1][len("refs/heads/") :].strip())
    return heads


def load_outbox_items(dirs: Sequence[Path]) -> tuple[list[dict], int]:
    """Load outbox item JSON files; unreadable files are counted, not hidden."""
    items: list[dict] = []
    unreadable = 0
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                item = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                unreadable += 1
                continue
            if isinstance(item, dict):
                item["_source"] = str(d)
                item["_file"] = f.name
                items.append(item)
            else:
                unreadable += 1
    return items, unreadable


# ---------------------------------------------------------------------------
# Pure computation
# ---------------------------------------------------------------------------


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_str(value: object) -> str | None:
    """Return value only when it is a non-empty string (real outbox data
    occasionally carries dicts/lists in these fields)."""
    return value if isinstance(value, str) and value else None


def _item_branch(item: dict) -> str | None:
    branch = _as_str(item.get("branch"))
    if not branch:
        evidence = item.get("local_evidence")
        if isinstance(evidence, dict):
            branch = _as_str(evidence.get("branch"))
    return branch


def _item_published(item: dict, pr_head_refs: set[str]) -> bool:
    if item.get("published") is True:
        return True
    publication = item.get("publication")
    if isinstance(publication, dict) and (
        publication.get("published") is True or publication.get("state") == "published"
    ):
        return True
    if item.get("pr_number") or item.get("pr_url"):
        return True
    if "already_satisfied" in str(item.get("requested_action", "")):
        return True
    branch = _item_branch(item)
    return bool(branch and branch in pr_head_refs)


def compute_work_loss(
    *,
    outbox_items: list[dict],
    remote_heads: set[str],
    prs: list[dict],
    now: datetime,
    window_start: datetime,
    unreadable_outbox_items: int = 0,
) -> dict:
    """Compute the waste report from already-loaded inputs."""
    pr_head_refs = {ref for pr in prs if (ref := _as_str(pr.get("head_ref")))}
    claimed: set[str] = set()
    lost_never_pushed: list[str] = []
    expired_unpublished: list[str] = []

    for item in outbox_items:
        branch = _item_branch(item)
        key = branch or _as_str(item.get("idempotency_key")) or _as_str(item.get("_file")) or ""
        if not key or key in claimed:
            continue
        if _item_published(item, pr_head_refs):
            continue
        expires_at = _parse_iso(item.get("expires_at"))
        if branch and branch not in remote_heads:
            # Never pushed wins over expired: the work exists only locally.
            lost_never_pushed.append(key)
            claimed.add(key)
            if branch:
                claimed.add(branch)
        elif expires_at and expires_at < now:
            expired_unpublished.append(key)
            claimed.add(key)
            if branch:
                claimed.add(branch)

    orphan_branches = sorted(
        b for b in remote_heads - PROTECTED_BRANCHES if b not in pr_head_refs and b not in claimed
    )
    claimed.update(orphan_branches)

    closed_unmerged = []
    produced = []
    for pr in prs:
        merged_at = _parse_iso(pr.get("merged_at"))
        if pr.get("merged") and merged_at and merged_at >= window_start:
            produced.append(pr["number"])
        elif pr.get("state") == "closed" and not pr.get("merged"):
            closed_at = _parse_iso(pr.get("closed_at"))
            if closed_at and closed_at >= window_start:
                closed_unmerged.append(pr["number"])

    lost_units = (
        len(lost_never_pushed)
        + len(expired_unpublished)
        + len(orphan_branches)
        + len(closed_unmerged)
    )
    produced_units = len(produced)
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "window_start": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_end": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branches_pushed_never_prd": len(orphan_branches),
        "outbox_expired_unpublished": len(expired_unpublished),
        "outbox_lost_never_pushed": len(lost_never_pushed),
        "prs_closed_unmerged": len(closed_unmerged),
        "produced_units": produced_units,
        "lost_units": lost_units,
        "waste_ratio": lost_units / max(1, produced_units),
        "outbox_items_scanned": len(outbox_items),
        "unreadable_outbox_items": unreadable_outbox_items,
        "remote_heads_scanned": len(remote_heads),
        "prs_scanned": len(prs),
        "sample_lost_never_pushed": lost_never_pushed[:10],
        "sample_expired_unpublished": expired_unpublished[:10],
        "sample_branches_never_prd": orphan_branches[:10],
        "unit_definitions": UNIT_DEFINITIONS,
    }


def render_waste_block(result: dict) -> str:
    """Render the waste markdown table from a compute_work_loss() result."""
    rows = [
        ("Window (produced/closed units)", f"{result['window_start']} -> {result['window_end']}"),
        ("Branches pushed, never PR'd", result["branches_pushed_never_prd"]),
        ("Outbox items expired unpublished", result["outbox_expired_unpublished"]),
        ("Outbox items lost, never pushed", result["outbox_lost_never_pushed"]),
        ("PRs closed unmerged (window)", result["prs_closed_unmerged"]),
        ("Lost units (deduplicated)", result["lost_units"]),
        ("Produced units (merged PRs in window)", result["produced_units"]),
        ("Waste ratio (lost_units / max(1, produced_units))", f"{result['waste_ratio']:.4g}"),
        (
            "Outbox items scanned / unreadable",
            f"{result['outbox_items_scanned']} / {result['unreadable_outbox_items']}",
        ),
        ("Methodology version", result["methodology_version"]),
    ]
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {k} | {v} |" for k, v in rows)
    lines.append("")
    lines.append("Unit definitions:")
    lines.extend(f"- `{k}`: {v}" for k, v in result["unit_definitions"].items())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--outbox-dir",
        action="append",
        default=None,
        help="Outbox dir(s) to scan (repeatable; default: live + archive).",
    )
    parser.add_argument(
        "--ls-remote-file",
        default=None,
        help="Captured `git ls-remote --heads origin` output (default: run git).",
    )
    parser.add_argument(
        "--prs-file",
        default=None,
        help="JSON list of PRs with number/head_ref/state/merged/merged_at/"
        "closed_at (default: fetch all PRs via gh REST).",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ISO8601 UTC window start override (default: now - window-days).",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--status-doc", default=DEFAULT_STATUS_DOC)
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    if args.since:
        window_start = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    else:
        window_start = now - timedelta(days=args.window_days)

    outbox_dirs = [Path(d) for d in (args.outbox_dir or DEFAULT_OUTBOX_DIRS)]
    outbox_items, unreadable = load_outbox_items(outbox_dirs)

    if args.ls_remote_file:
        ls_remote_text = Path(args.ls_remote_file).read_text()
    else:
        ls_remote_text = run_ls_remote()
    remote_heads = parse_ls_remote(ls_remote_text)

    if args.prs_file:
        prs = json.loads(Path(args.prs_file).read_text())
    else:
        prs = fetch_all_prs(args.repo)

    result = compute_work_loss(
        outbox_items=outbox_items,
        remote_heads=remote_heads,
        prs=prs,
        now=now,
        window_start=window_start,
        unreadable_outbox_items=unreadable,
    )

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"waste_ratio={result['waste_ratio']:.4g} "
            f"(lost={result['lost_units']} "
            f"[never-pushed={result['outbox_lost_never_pushed']}, "
            f"expired={result['outbox_expired_unpublished']}, "
            f"orphan-branches={result['branches_pushed_never_prd']}, "
            f"closed-unmerged={result['prs_closed_unmerged']}] / "
            f"produced={result['produced_units']})"
        )

    if args.publish:
        doc = Path(args.status_doc)
        update_leverage_md(doc, waste_block=render_waste_block(result))
        print(f"published waste section to {doc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

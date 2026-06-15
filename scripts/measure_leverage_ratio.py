#!/usr/bin/env python3
"""Measure the steering-program Leverage Ratio (LR).

Sprint 4 goal 2 / Steering Leverage Operating Plan v2, Phase 0.2.

LR = verified merged outcomes / operator-minutes, per window.

A "verified outcome" is a PR merged inside the window whose body or comments
reference a DecisionReceipt path that exists locally AND passes
``aragora receipt verify``. Receipts that exist but fail verification are
counted separately (``receipts_failed_verify``) and never silently dropped.

Contract guarantees (see the program doc):
- ``--operator-minutes`` is REQUIRED. The script refuses to run without it
  (exit 2): the number is operator-estimated until attention capture exists,
  and this script will never invent it.
- ``steering_integrity`` is published as ``null`` — never a number — until its
  three components (crux_shown, within_attention_budget, not_reversed_on_audit)
  are actually instrumented.

``--publish`` renders/updates ``docs/status/LEVERAGE.md`` inside a
marker-delimited managed region; text outside the region (e.g. the manual
Blind-Period Log entries) is never touched.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

METHODOLOGY_VERSION = 1
DEFAULT_REPO = "synaptent/aragora"
DEFAULT_RECEIPTS_DIRS = (".aragora/run-20260610/receipts",)
DEFAULT_STATUS_DOC = "docs/status/LEVERAGE.md"

SI_COMPONENTS_PENDING = [
    "crux_shown",
    "within_attention_budget",
    "not_reversed_on_audit",
]

OPERATOR_MINUTES_REFUSAL = (
    "refusing to run: --operator-minutes is required and must be > 0.\n"
    "Operator-minutes are operator-estimated (self-reported) until attention "
    "capture exists; this script will never invent the number. Estimate the "
    "human minutes actually spent steering this window (approvals, design "
    "reviews, queue replies) and pass it explicitly."
)

# Receipt references in PR bodies/comments: any path-ish token whose parent
# directory chain contains a `receipts/` segment and that ends in .json.
RECEIPT_REF_RE = re.compile(r"[\w@~./\-]*receipts/[\w.@\-]+\.json")


# ---------------------------------------------------------------------------
# Subprocess boundaries (mockable)
# ---------------------------------------------------------------------------


def _run_gh(args: list[str]) -> str:
    """Run a gh CLI command and return stdout (raises on failure)."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=True, timeout=300)
    return proc.stdout


def fetch_merged_prs(repo: str, since: datetime) -> list[dict]:
    """Fetch PRs merged at/after ``since`` via the GitHub search REST API."""
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    query = f"repo:{repo} is:pr is:merged merged:>={since_str}"
    out = _run_gh(
        [
            "api",
            "--paginate",
            "-X",
            "GET",
            "search/issues",
            "-f",
            f"q={query}",
            "--jq",
            ".items[] | {number: .number, title: .title, body: .body, "
            "merged_at: .pull_request.merged_at}",
        ]
    )
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def fetch_issue_comments(repo: str, number: int) -> list[str]:
    """Fetch issue-comment bodies for a PR."""
    out = _run_gh(
        [
            "api",
            "--paginate",
            f"repos/{repo}/issues/{number}/comments",
            "--jq",
            ".[].body",
        ]
    )
    # --jq emits raw strings per line; comments may be multi-line, so fetch
    # as JSON instead when precision matters. For receipt-path scanning,
    # newline-joined text is sufficient.
    return [out] if out.strip() else []


def verify_receipt(path: Path) -> bool:
    """Verify a receipt via ``aragora receipt verify`` (subprocess boundary)."""
    try:
        proc = subprocess.run(
            ["aragora", "receipt", "verify", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def find_receipt_refs(text: str) -> list[str]:
    """Extract unique receipt-path references from free text, in order."""
    seen: dict[str, None] = {}
    for match in RECEIPT_REF_RE.finditer(text or ""):
        seen.setdefault(match.group(0), None)
    return list(seen)


def resolve_receipt_path(ref: str, receipts_dirs: Sequence[Path]) -> Path | None:
    """Resolve a referenced receipt path to a local file, or None."""
    candidates = [Path(ref).expanduser()]
    basename = Path(ref).name
    candidates.extend(Path(d) / basename for d in receipts_dirs)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def compute_leverage(
    *,
    merged_prs: list[dict],
    operator_minutes: float,
    receipts_dirs: Sequence[Path],
    comments_fetcher: Callable[[int], list[str]],
    verifier: Callable[[Path], bool],
    window_start: datetime,
    window_end: datetime,
    window_days: int,
    repo: str,
    operator_minutes_note: str = "",
) -> dict:
    """Compute the leverage-ratio report from already-fetched inputs."""
    merged_receipt_backed = 0
    failed_verify_paths: list[str] = []
    refs_unresolved = 0
    verified_cache: dict[Path, bool] = {}
    backed_prs: list[int] = []
    unique_receipts: set[str] = set()

    for pr in merged_prs:
        texts = [pr.get("body") or ""]
        texts.extend(comments_fetcher(pr["number"]))
        refs = find_receipt_refs("\n".join(texts))
        backed = False
        for ref in refs:
            path = resolve_receipt_path(ref, receipts_dirs)
            if path is None:
                refs_unresolved += 1
                continue
            if path not in verified_cache:
                verified_cache[path] = verifier(path)
                if not verified_cache[path]:
                    failed_verify_paths.append(str(path))
            if verified_cache[path]:
                backed = True
                unique_receipts.add(str(path))
        if backed:
            merged_receipt_backed += 1
            backed_prs.append(pr["number"])

    # Anti-gaming guard: splitting one piece of work across N PRs that all
    # cite the same receipt inflates merged_receipt_backed but not
    # unique_receipts_backed; split_factor > 1 surfaces the divergence.
    unique_receipts_backed = len(unique_receipts)
    split_factor = merged_receipt_backed / unique_receipts_backed if unique_receipts_backed else 0.0

    return {
        "methodology_version": METHODOLOGY_VERSION,
        "repo": repo,
        "window_days": window_days,
        "window_start": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_end": window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operator_minutes": operator_minutes,
        "operator_minutes_source": ("self-reported (operator-estimated; no attention capture yet)"),
        "operator_minutes_note": operator_minutes_note,
        "merged_total": len(merged_prs),
        "merged_receipt_backed": merged_receipt_backed,
        "unique_receipts_backed": unique_receipts_backed,
        "split_factor": split_factor,
        "receipt_backed_pr_numbers": backed_prs,
        "receipts_failed_verify": len(failed_verify_paths),
        "failed_verify_paths": failed_verify_paths,
        "receipt_refs_unresolved": refs_unresolved,
        "leverage_ratio": merged_receipt_backed / operator_minutes,
        "steering_integrity": None,
        "si_components_pending": SI_COMPONENTS_PENDING,
    }


# ---------------------------------------------------------------------------
# LEVERAGE.md managed-region rendering
# ---------------------------------------------------------------------------

MANAGED_BEGIN = "<!-- leverage-managed:begin -->"
MANAGED_END = "<!-- leverage-managed:end -->"
LR_BEGIN = "<!-- leverage-lr:begin -->"
LR_END = "<!-- leverage-lr:end -->"
WASTE_BEGIN = "<!-- leverage-waste:begin -->"
WASTE_END = "<!-- leverage-waste:end -->"
LR_PLACEHOLDER = "_Leverage ratio not yet measured._"
WASTE_PLACEHOLDER = "_Waste ratio not yet measured._"

CAVEATS = """\
## Caveats (honest limits of these numbers)

- **Operator-minutes are self-reported.** There is no attention capture yet;
  the denominator is an operator estimate passed explicitly on the CLI, and
  the script refuses to run without it.
- **Steering Integrity (SI) is not yet instrumented.** It is published as
  `null` — never a number — until crux_shown, within_attention_budget, and
  not_reversed_on_audit are actually captured.
- **Receipt linkage is text-based.** A PR counts as receipt-backed when its
  body/comments reference a receipt path that exists locally and verifies;
  this can undercount (receipts not referenced) or be gamed by splitting work
  into more PRs — merged_total, unique_receipts_backed, and split_factor are
  published alongside so splitting is visible, not hidden.
- **Waste units are defined in the waste table above**; categories are
  de-duplicated by unit key (branch name, else outbox idempotency key) so a
  unit of lost work is counted at most once.
"""

BLIND_PERIOD_SEED = """\
## Blind-Period Log

Manual entries below are preserved across re-renders; the publisher never
touches text outside the managed region above.

- 2026-05-27 -> 2026-06-05: automation publisher dead / loop blind
  (source: `.aragora/run-20260610/OPERATOR_STEERING_AUDIT.md`).
"""


def _extract_block(text: str, begin: str, end: str) -> str | None:
    try:
        start = text.index(begin) + len(begin)
        stop = text.index(end, start)
    except ValueError:
        return None
    return text[start:stop].strip("\n")


def _render_managed(lr_block: str, waste_block: str, now_iso: str) -> str:
    return (
        f"{MANAGED_BEGIN}\n"
        f"Last updated: {now_iso}\n\n"
        "Repo-tracked recurring publication surface for the steering-leverage "
        "program\n(Operating Plan v2, Phase 0.2): leverage ratio (LR) and "
        "waste ratio together.\n\n"
        "## Leverage Ratio (LR)\n\n"
        f"{LR_BEGIN}\n{lr_block}\n{LR_END}\n\n"
        "## Waste Ratio (Work-Loss Accounting)\n\n"
        f"{WASTE_BEGIN}\n{waste_block}\n{WASTE_END}\n\n"
        f"{CAVEATS}"
        f"{MANAGED_END}\n"
    )


def update_leverage_md(
    path: Path,
    *,
    lr_block: str | None = None,
    waste_block: str | None = None,
    now: datetime | None = None,
) -> str:
    """Create/update the managed region of LEVERAGE.md.

    Only the managed region between the markers is rewritten. Whichever of
    ``lr_block`` / ``waste_block`` is not supplied is carried over from the
    existing file (or a placeholder on first render). Text outside the
    managed region is never modified.
    """
    now_iso = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing = path.read_text() if path.exists() else ""

    prior_lr = _extract_block(existing, LR_BEGIN, LR_END)
    prior_waste = _extract_block(existing, WASTE_BEGIN, WASTE_END)
    managed = _render_managed(
        lr_block if lr_block is not None else (prior_lr or LR_PLACEHOLDER),
        waste_block if waste_block is not None else (prior_waste or WASTE_PLACEHOLDER),
        now_iso,
    )

    if MANAGED_BEGIN in existing and MANAGED_END in existing:
        start = existing.index(MANAGED_BEGIN)
        stop = existing.index(MANAGED_END) + len(MANAGED_END)
        new_text = existing[:start] + managed.rstrip("\n") + existing[stop:]
        if not new_text.endswith("\n"):
            new_text += "\n"
    elif existing:
        new_text = existing.rstrip("\n") + "\n\n" + managed
    else:
        new_text = "# Leverage & Waste Status\n\n" + managed + "\n" + BLIND_PERIOD_SEED

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text)
    return new_text


def render_lr_block(result: dict) -> str:
    """Render the LR markdown table from a compute_leverage() result."""
    si_pending = ", ".join(result["si_components_pending"])
    rows = [
        (
            "Window",
            f"{result['window_start']} -> {result['window_end']} ({result['window_days']}d)",
        ),
        ("Merged PRs in window (total)", result["merged_total"]),
        ("Merged PRs receipt-backed (verified)", result["merged_receipt_backed"]),
        ("Unique verified receipts backing them", result["unique_receipts_backed"]),
        (
            "Split factor (receipt-backed PRs / unique receipts; >1 = splitting)",
            f"{result['split_factor']:.4g}",
        ),
        ("Receipts failed verify", result["receipts_failed_verify"]),
        ("Receipt refs unresolved locally", result["receipt_refs_unresolved"]),
        ("Operator minutes (self-reported)", result["operator_minutes"]),
        (
            "Leverage ratio (verified merged outcomes / operator-minute)",
            f"{result['leverage_ratio']:.4g}",
        ),
        ("Steering integrity (SI)", f"null — pending: {si_pending}"),
        ("Methodology version", result["methodology_version"]),
    ]
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {k} | {v} |" for k, v in rows)
    if result.get("operator_minutes_note"):
        lines.append("")
        lines.append(f"Operator-minutes note: {result['operator_minutes_note']}")
    if result.get("failed_verify_paths"):
        lines.append("")
        lines.append("Receipts that failed verification (never silent):")
        lines.extend(f"- `{p}`" for p in result["failed_verify_paths"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--operator-minutes",
        type=float,
        default=None,
        help="REQUIRED. Self-reported operator minutes for the window; the "
        "script refuses to invent this number.",
    )
    parser.add_argument(
        "--operator-minutes-note",
        default="",
        help="Free-text provenance for the operator-minutes estimate.",
    )
    parser.add_argument(
        "--receipts-dir",
        action="append",
        default=None,
        help="Local receipts dir(s) used to resolve referenced receipt paths "
        f"(repeatable; default: {DEFAULT_RECEIPTS_DIRS[0]}).",
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

    if args.operator_minutes is None or args.operator_minutes <= 0:
        print(OPERATOR_MINUTES_REFUSAL, file=sys.stderr)
        return 2

    window_end = datetime.now(timezone.utc)
    if args.since:
        window_start = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    else:
        window_start = window_end - timedelta(days=args.window_days)

    receipts_dirs = [Path(d) for d in (args.receipts_dir or DEFAULT_RECEIPTS_DIRS)]
    merged_prs = fetch_merged_prs(args.repo, window_start)
    result = compute_leverage(
        merged_prs=merged_prs,
        operator_minutes=args.operator_minutes,
        receipts_dirs=receipts_dirs,
        comments_fetcher=lambda n: fetch_issue_comments(args.repo, n),
        verifier=verify_receipt,
        window_start=window_start,
        window_end=window_end,
        window_days=args.window_days,
        repo=args.repo,
        operator_minutes_note=args.operator_minutes_note,
    )

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"LR={result['leverage_ratio']:.4g} "
            f"({result['merged_receipt_backed']}/{result['merged_total']} merged "
            f"PRs receipt-backed; {result['receipts_failed_verify']} failed "
            f"verify; {result['operator_minutes']} operator-min; SI=null)"
        )

    if args.publish:
        doc = Path(args.status_doc)
        update_leverage_md(doc, lr_block=render_lr_block(result))
        print(f"published LR section to {doc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""H1-01 dry-run dispatch protocol for the rev-4 staging corpus.

This script produces the recorded ``worker_outcome`` dispatch evidence that
``docs/benchmarks/corpus_rev4_staging.md`` requires before staging entries
can be promoted to ``docs/benchmarks/corpus.json``. It does so safely:

- It does not open PRs or push branches.
- It does not call any LLM provider.
- It does not modify the canonical rev-3 corpus or the rev-3 freshness
  invariant.
- It records ``worker_status="dry_run"`` and ``worker_outcome="dry_run_*"``
  rows so the dispatch evidence is unambiguous.

What the script does:

1. Reads ``tests/benchmarks/corpus_rev4.json`` (33 staging entries).
2. For each entry: pulls the GitHub issue title + body via ``gh issue view``,
   runs it through ``aragora.swarm.task_sanitizer.TaskSanitizer.sanitize``,
   classifies the outcome, and appends a single boss-metrics row to a
   *parallel* dispatch ledger at
   ``.aragora/overnight/boss_metrics_h1_01_dry_run.jsonl`` (NOT the
   production ledger).
3. Writes a per-issue summary JSON at
   ``.aragora/evolve-round/2026-04-30b/dogfood/h1-01-dry-run-summary.json``
   and a Markdown report at the sibling ``.md`` path.

Usage::

    python3 scripts/h1_01_dry_run_dispatch.py
    python3 scripts/h1_01_dry_run_dispatch.py --limit 5  # smoke test
    python3 scripts/h1_01_dry_run_dispatch.py --json     # JSON to stdout
    python3 scripts/h1_01_dry_run_dispatch.py --offline  # skip gh calls

The script is idempotent: re-running it appends new rows but the per-issue
summary always overwrites with the latest dispatch result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_REV4_PATH = REPO_ROOT / "tests/benchmarks/corpus_rev4.json"
OVERNIGHT_DIR = REPO_ROOT / ".aragora/overnight"
DRY_RUN_LEDGER = OVERNIGHT_DIR / "boss_metrics_h1_01_dry_run.jsonl"
ROUND_DIR = REPO_ROOT / ".aragora/evolve-round/2026-04-30b/dogfood"
SUMMARY_JSON = ROUND_DIR / "h1-01-dry-run-summary.json"
SUMMARY_MD = ROUND_DIR / "h1-01-dry-run-summary.md"


@dataclass(slots=True)
class DispatchRow:
    issue_id: int
    title: str
    execution_class: str
    sanitizer_outcome: str
    sanitizer_reason: str
    sanitizer_checks_failed: list[str]
    sanitizer_confidence: float
    body_chars: int
    sanitized_chars: int
    fetch_status: str
    terminal_class: str


def _load_corpus(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gh_issue_view(issue_id: int) -> tuple[str, str, str]:
    """Return (title, body, fetch_status). fetch_status='ok'|'offline'|'error'."""
    try:
        proc = subprocess.run(
            ["gh", "issue", "view", str(issue_id), "--json", "title,body,state"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return "", "", "offline:gh-not-found"
    except subprocess.TimeoutExpired:
        return "", "", "offline:timeout"
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else f"rc={proc.returncode}"
        return "", "", f"error:{detail[:80]}"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return "", "", "error:non-json"
    return str(data.get("title") or ""), str(data.get("body") or ""), "ok"


def _classify_terminal(outcome: str) -> str:
    """Map sanitizer outcome to a canonical terminal-class bucket.

    These names mirror the canonical taxonomy in
    ``aragora/swarm/boss_loop_outcome.py``. Dry-run rows use a clearly
    namespaced ``dry_run_*`` prefix so they are never confused with real
    boss-loop terminal classes by downstream consumers.
    """
    if outcome == "accepted":
        return "dry_run_accepted"
    if outcome == "rewritten":
        return "dry_run_rewritten"
    if outcome == "dropped":
        return "dry_run_dropped"
    if outcome == "quarantined":
        return "dry_run_quarantined"
    return "dry_run_unknown"


def _row_for_issue(issue_id: int, exec_class: str, *, offline: bool) -> DispatchRow:
    if offline:
        title, body, fetch_status = "", "", "offline:requested"
    else:
        title, body, fetch_status = _gh_issue_view(issue_id)
    if fetch_status != "ok":
        return DispatchRow(
            issue_id=issue_id,
            title="",
            execution_class=exec_class,
            sanitizer_outcome="skipped",
            sanitizer_reason=f"fetch failed: {fetch_status}",
            sanitizer_checks_failed=[],
            sanitizer_confidence=0.0,
            body_chars=0,
            sanitized_chars=0,
            fetch_status=fetch_status,
            terminal_class="dry_run_skipped",
        )

    from aragora.swarm.task_sanitizer import TaskSanitizer

    sanitizer = TaskSanitizer(repo_root=REPO_ROOT)
    result = sanitizer.sanitize(title, body)
    return DispatchRow(
        issue_id=issue_id,
        title=title[:120],
        execution_class=exec_class,
        sanitizer_outcome=result.outcome.value,
        sanitizer_reason=result.reason[:200],
        sanitizer_checks_failed=list(result.checks_failed),
        sanitizer_confidence=float(result.confidence),
        body_chars=len(body),
        sanitized_chars=len(result.sanitized_text),
        fetch_status=fetch_status,
        terminal_class=_classify_terminal(result.outcome.value),
    )


def _persist_metrics_row(row: DispatchRow, *, ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue_number": row.issue_id,
        "issue_title": row.title,
        "iteration": 1,
        "elapsed_seconds": 0.0,
        "prompt_chars": row.body_chars,
        "prompt_version": "h1_01_dry_run_v1",
        "sanitizer_outcome": row.sanitizer_outcome,
        "sanitizer_checks_failed_count": len(row.sanitizer_checks_failed),
        "tests_run": 0,
        "tests_passed": 0,
        "files_changed": 0,
        "publish_action": "dry_run_no_publish",
        "worker_status": "dry_run",
        "worker_outcome": f"dry_run:{row.sanitizer_outcome}",
        "failure_reason": None if row.fetch_status == "ok" else row.fetch_status,
        "terminal_class": row.terminal_class,
        "blocker_kind": None,
        "blocker_evidence": None,
        "category_success_rates": {},
        "cohort_tag": "h1-01-dry-run-evidence",
        "deferred_queue_depth": 0,
        "enriched_context_chars": row.sanitized_chars,
        "has_deliverable": False,
        "is_decomposed_issue": False,
        "execution_class": row.execution_class,
        "recorded_at": datetime.now(tz=UTC).isoformat(),
    }
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _summarize(rows: list[DispatchRow]) -> dict:
    by_class: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    by_terminal: dict[str, int] = {}
    fetch_ok = 0
    for row in rows:
        by_class[row.execution_class] = by_class.get(row.execution_class, 0) + 1
        by_outcome[row.sanitizer_outcome] = by_outcome.get(row.sanitizer_outcome, 0) + 1
        by_terminal[row.terminal_class] = by_terminal.get(row.terminal_class, 0) + 1
        if row.fetch_status == "ok":
            fetch_ok += 1
    return {
        "total": len(rows),
        "fetch_ok": fetch_ok,
        "by_execution_class": dict(sorted(by_class.items())),
        "by_sanitizer_outcome": dict(sorted(by_outcome.items())),
        "by_terminal_class": dict(sorted(by_terminal.items())),
    }


def _write_markdown(rows: list[DispatchRow], summary: dict) -> str:
    lines = [
        "# H1-01 dry-run dispatch summary",
        "",
        "Generated by `scripts/h1_01_dry_run_dispatch.py`. This is the recorded",
        "dispatch evidence required by `docs/benchmarks/corpus_rev4_staging.md`",
        "before promoting rev-4 staging to canonical.",
        "",
        "## Aggregate",
        "",
        f"- Total entries dispatched: **{summary['total']}**",
        f"- Successful gh fetch: **{summary['fetch_ok']}**",
        "",
        "### By execution class",
        "",
    ]
    for cls, n in summary["by_execution_class"].items():
        lines.append(f"- `{cls}` — {n}")
    lines.extend(["", "### By sanitizer outcome", ""])
    for outc, n in summary["by_sanitizer_outcome"].items():
        lines.append(f"- `{outc}` — {n}")
    lines.extend(["", "## Per-issue rows", "", "| issue | class | outcome | fetch |"])
    lines.append("|---:|---|---|---|")
    for r in rows:
        lines.append(
            f"| #{r.issue_id} | {r.execution_class} | {r.sanitizer_outcome} | {r.fetch_status} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="dispatch only the first N entries")
    parser.add_argument("--json", action="store_true", help="emit summary as JSON to stdout")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip gh fetches; mark every row as offline:requested (smoke test)",
    )
    parser.add_argument(
        "--ledger-path",
        default=str(DRY_RUN_LEDGER),
        help="override the dry-run dispatch ledger path",
    )
    args = parser.parse_args(argv)

    corpus = _load_corpus(CORPUS_REV4_PATH)
    issues = corpus.get("issues", [])
    if args.limit is not None:
        issues = issues[: args.limit]

    rows: list[DispatchRow] = []
    ledger_path = Path(args.ledger_path)
    for entry in issues:
        issue_id = int(entry["issue_id"])
        exec_class = str(entry.get("execution_class") or "unknown")
        row = _row_for_issue(issue_id, exec_class, offline=args.offline)
        rows.append(row)
        _persist_metrics_row(row, ledger_path=ledger_path)

    summary = _summarize(rows)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(
        json.dumps(
            {"summary": summary, "rows": [asdict(r) for r in rows]},
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    SUMMARY_MD.write_text(_write_markdown(rows, summary), encoding="utf-8")

    if args.json:
        print(json.dumps({"summary": summary}, sort_keys=True, indent=2))
    else:
        print(
            f"h1-01 dry-run: dispatched {summary['total']} entries "
            f"(fetch_ok={summary['fetch_ok']}); ledger={ledger_path}; "
            f"summary={SUMMARY_MD}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

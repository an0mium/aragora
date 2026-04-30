#!/usr/bin/env python3
"""H1-02 advisory zero-rescue scorecard for the rev-4 staging corpus.

This script is the H1-02 deliverable that operates *before* canonical
promotion. It reads:

1. The rev-4 staging manifest at ``tests/benchmarks/corpus_rev4.json``.
2. The dry-run dispatch ledger at
   ``.aragora/overnight/boss_metrics_h1_01_dry_run.jsonl`` produced by
   PR #6828's ``scripts/h1_01_dry_run_dispatch.py``.

It computes a *staging* zero-rescue scorecard and persists it under
``.aragora/evolve-round/2026-04-30b/dogfood/h1-02-rev4-staging-scorecard.json``
with a human-readable Markdown sibling.

Scorecard semantics (advisory-only):

- ``dispatched_count``: number of staging entries with at least one
  ``dry_run:*`` outcome row in the ledger.
- ``accepted_rate``: dispatched-and-accepted / total-staging.
- ``promotion_floor_met``: true when ``dispatched_count >= 15`` per
  ``corpus_rev4_staging.md``.
- ``per_class``: dispatched/accepted/dropped/quarantined counts per
  execution class.

This is **not** a replacement for the canonical
``build_benchmark_truth_artifact.py``: that script measures the rev-3
canonical corpus against real PR-truth on GitHub. This script measures
staging dispatch evidence only and uses the ``dry_run`` namespace
reserved by PR #6828, so its output cannot be confused with canonical
benchmark truth.

Usage::

    python3 scripts/h1_02_rev4_staging_scorecard.py
    python3 scripts/h1_02_rev4_staging_scorecard.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_REV4_PATH = REPO_ROOT / "tests/benchmarks/corpus_rev4.json"
DRY_RUN_LEDGER = REPO_ROOT / ".aragora/overnight/boss_metrics_h1_01_dry_run.jsonl"
ROUND_DIR = REPO_ROOT / ".aragora/evolve-round/2026-04-30b/dogfood"
SCORECARD_JSON = ROUND_DIR / "h1-02-rev4-staging-scorecard.json"
SCORECARD_MD = ROUND_DIR / "h1-02-rev4-staging-scorecard.md"

PROMOTION_FLOOR = 15


@dataclass(slots=True)
class ClassBreakdown:
    total: int = 0
    dispatched: int = 0
    accepted: int = 0
    rewritten: int = 0
    dropped: int = 0
    quarantined: int = 0
    skipped: int = 0
    unknown: int = 0


@dataclass(slots=True)
class Scorecard:
    rev: int
    status: str
    total_staging: int
    dispatched_count: int
    accepted_count: int
    promotion_floor: int
    promotion_floor_met: bool
    accepted_rate: float
    per_class: dict[str, ClassBreakdown] = field(default_factory=dict)
    generated_at: str = ""
    ledger_path: str = ""
    corpus_path: str = ""


def _load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _outcome_for_issue(rows: list[dict[str, Any]]) -> str:
    """Pick the latest dry-run outcome for an issue.

    Each issue may have multiple rows (re-runs). The latest non-skipped
    outcome wins; if all are skipped, ``skipped`` is reported.
    """
    latest = "skipped"
    for r in rows:
        wo = str(r.get("worker_outcome") or "")
        if wo.startswith("dry_run:"):
            outcome = wo.split(":", 1)[1] or "unknown"
            if outcome != "skipped":
                latest = outcome
    if latest == "skipped":
        for r in reversed(rows):
            wo = str(r.get("worker_outcome") or "")
            if wo.startswith("dry_run:"):
                latest = wo.split(":", 1)[1] or "unknown"
                break
    return latest


def compute_scorecard(
    corpus: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
) -> Scorecard:
    issues = corpus.get("issues") or []
    total = len(issues)

    rows_by_issue: dict[int, list[dict[str, Any]]] = {}
    for r in ledger_rows:
        n = r.get("issue_number")
        if isinstance(n, int) and n > 0:
            rows_by_issue.setdefault(n, []).append(r)

    per_class: dict[str, ClassBreakdown] = {}
    dispatched_count = 0
    accepted_count = 0

    for entry in issues:
        cls = str(entry.get("execution_class") or "unknown")
        b = per_class.setdefault(cls, ClassBreakdown())
        b.total += 1
        issue_id = int(entry.get("issue_id") or 0)
        if issue_id <= 0 or issue_id not in rows_by_issue:
            continue
        b.dispatched += 1
        dispatched_count += 1
        outcome = _outcome_for_issue(rows_by_issue[issue_id])
        if outcome == "accepted":
            b.accepted += 1
            accepted_count += 1
        elif outcome == "rewritten":
            b.rewritten += 1
        elif outcome == "dropped":
            b.dropped += 1
        elif outcome == "quarantined":
            b.quarantined += 1
        elif outcome == "skipped":
            b.skipped += 1
        else:
            b.unknown += 1

    accepted_rate = (accepted_count / total) if total else 0.0
    return Scorecard(
        rev=int(corpus.get("revision") or 4),
        status=str(corpus.get("status") or "unknown"),
        total_staging=total,
        dispatched_count=dispatched_count,
        accepted_count=accepted_count,
        promotion_floor=PROMOTION_FLOOR,
        promotion_floor_met=dispatched_count >= PROMOTION_FLOOR,
        accepted_rate=accepted_rate,
        per_class=per_class,
        generated_at=datetime.now(tz=UTC).isoformat(),
        ledger_path=str(DRY_RUN_LEDGER),
        corpus_path=str(CORPUS_REV4_PATH),
    )


def render_markdown(card: Scorecard) -> str:
    floor_emoji = "OK" if card.promotion_floor_met else "PENDING"
    lines = [
        "# H1-02 rev-4 staging scorecard (advisory)",
        "",
        f"_Generated: {card.generated_at}_",
        "",
        "## Aggregate",
        "",
        f"- Corpus rev: **{card.rev}**, status: **{card.status}**",
        f"- Staging entries: **{card.total_staging}**",
        f"- Dispatched (≥1 dry-run row): **{card.dispatched_count}**",
        f"- Accepted on dispatch: **{card.accepted_count}**",
        f"- Accepted rate: **{card.accepted_rate:.1%}**",
        f"- Promotion floor: **{card.promotion_floor}** "
        f"({floor_emoji}: {card.dispatched_count}/{card.promotion_floor})",
        "",
        "## Per execution class",
        "",
        "| class | total | dispatched | accepted | rewritten | dropped | quarantined | skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cls, b in sorted(card.per_class.items()):
        lines.append(
            f"| {cls} | {b.total} | {b.dispatched} | {b.accepted} | "
            f"{b.rewritten} | {b.dropped} | {b.quarantined} | {b.skipped} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Ledger: `{card.ledger_path}`",
            f"- Corpus: `{card.corpus_path}`",
            "",
            "## Status",
            "",
        ]
    )
    if card.promotion_floor_met:
        lines.append(
            "Promotion floor is met. The next operator may open a follow-up "
            "PR to migrate the dispatched entries into "
            "`docs/benchmarks/corpus.json` per the rev-4 staging promotion "
            "contract."
        )
    else:
        lines.append(
            f"Promotion floor not yet met. Need "
            f"{card.promotion_floor - card.dispatched_count} more dispatched "
            f"entries before promotion can proceed."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit scorecard as JSON to stdout")
    parser.add_argument(
        "--corpus-path",
        default=str(CORPUS_REV4_PATH),
        help="override the rev-4 staging corpus path",
    )
    parser.add_argument(
        "--ledger-path",
        default=str(DRY_RUN_LEDGER),
        help="override the dry-run dispatch ledger path",
    )
    parser.add_argument(
        "--scorecard-json-path",
        default=str(SCORECARD_JSON),
        help="override where the JSON scorecard is persisted",
    )
    parser.add_argument(
        "--scorecard-md-path",
        default=str(SCORECARD_MD),
        help="override where the Markdown scorecard is persisted",
    )
    args = parser.parse_args(argv)

    corpus = _load_corpus(Path(args.corpus_path))
    ledger_rows = _load_ledger(Path(args.ledger_path))
    card = compute_scorecard(corpus, ledger_rows)

    json_path = Path(args.scorecard_json_path)
    md_path = Path(args.scorecard_md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scorecard": asdict(card),
    }
    json_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(card), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2))
    else:
        floor_status = "MET" if card.promotion_floor_met else "PENDING"
        print(
            f"h1-02 rev-4 staging scorecard: rev={card.rev} status={card.status} "
            f"dispatched={card.dispatched_count}/{card.total_staging} "
            f"accepted={card.accepted_count} accepted_rate={card.accepted_rate:.1%} "
            f"promotion_floor={card.promotion_floor} ({floor_status})"
        )
        print(f"  json: {json_path}")
        print(f"  md:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

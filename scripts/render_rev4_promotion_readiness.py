#!/usr/bin/env python3
"""Render H1-01 rev-4 benchmark promotion readiness from local boss metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.utils.git_paths import git_common_repo_root, resolve_repo_fallback_path  # noqa: E402

DEFAULT_CORPUS_PATH = REPO_ROOT / "tests" / "benchmarks" / "corpus_rev4.json"
DEFAULT_METRICS_PATH = REPO_ROOT / ".aragora" / "overnight" / "boss_metrics.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "status" / "H1_01_REV4_PROMOTION_READINESS.md"
DEFAULT_MIN_DISPATCHED = 15


def _repo_stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _normalize_generated_at(value: str | None = None) -> str:
    if value:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = dt.datetime.now(dt.UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_metrics_path(candidate: Path) -> Path:
    return resolve_repo_fallback_path(
        candidate,
        repo_root=REPO_ROOT,
        common_root=git_common_repo_root(REPO_ROOT),
    )


def load_corpus(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Corpus at {path} must be a JSON object")
    issues = payload.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError(f"Corpus at {path} must contain a non-empty issues list")
    return payload


def load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _issue_id(issue: dict[str, Any]) -> int:
    return int(issue.get("issue_id", 0) or 0)


def build_readiness(
    *,
    corpus: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    min_dispatched: int = DEFAULT_MIN_DISPATCHED,
) -> dict[str, Any]:
    issues = [
        dict(issue)
        for issue in corpus.get("issues", [])
        if isinstance(issue, dict) and _issue_id(issue) > 0
    ]
    issues.sort(key=_issue_id)
    issue_ids = {_issue_id(issue) for issue in issues}
    rows_by_issue: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        issue_number = row.get("issue_number")
        if isinstance(issue_number, int) and issue_number in issue_ids:
            rows_by_issue[issue_number].append(row)

    dispatched_ids = sorted(rows_by_issue)
    missing_ids = sorted(issue_ids - set(dispatched_ids))
    needed_for_minimum = max(int(min_dispatched) - len(dispatched_ids), 0)
    recommended_ids = missing_ids[: needed_for_minimum or min(10, len(missing_ids))]

    class_totals = Counter(str(issue.get("execution_class") or "unknown") for issue in issues)
    class_dispatched: Counter[str] = Counter()
    latest_terminal_by_issue: dict[int, str] = {}
    for issue in issues:
        issue_number = _issue_id(issue)
        execution_class = str(issue.get("execution_class") or "unknown")
        rows = rows_by_issue.get(issue_number, [])
        if rows:
            class_dispatched[execution_class] += 1
        latest_terminal = ""
        for row in rows:
            terminal_class = str(row.get("terminal_class") or "").strip()
            if terminal_class:
                latest_terminal = terminal_class
        if latest_terminal:
            latest_terminal_by_issue[issue_number] = latest_terminal

    status = "promotion_ready" if needed_for_minimum == 0 else "needs_more_dispatch_evidence"
    if len(issues) < 30:
        status = "manifest_below_h1_floor"

    return {
        "status": status,
        "min_dispatched_for_first_slice": int(min_dispatched),
        "needed_for_minimum": needed_for_minimum,
        "corpus": {
            "corpus_id": str(corpus.get("corpus_id") or "").strip(),
            "revision": int(corpus.get("revision", 0) or 0),
            "status": str(corpus.get("status") or "").strip(),
            "recorded_on": str(corpus.get("recorded_on") or "").strip(),
            "issue_count": len(issues),
            "promotion_target": str(corpus.get("promotion_target") or "").strip(),
        },
        "dispatch": {
            "dispatched_issue_count": len(dispatched_ids),
            "missing_issue_count": len(missing_ids),
            "dispatched_issue_ids": dispatched_ids,
            "missing_issue_ids": missing_ids,
            "recommended_next_issue_ids": recommended_ids,
            "latest_terminal_by_issue": latest_terminal_by_issue,
        },
        "execution_classes": {
            execution_class: {
                "total": class_totals[execution_class],
                "dispatched": class_dispatched[execution_class],
                "missing": class_totals[execution_class] - class_dispatched[execution_class],
            }
            for execution_class in sorted(class_totals)
        },
    }


def _format_issue_ids(issue_ids: list[int]) -> str:
    if not issue_ids:
        return "none"
    return ", ".join(f"`#{issue_id}`" for issue_id in issue_ids)


def render_markdown(
    *,
    readiness: dict[str, Any],
    corpus_path: Path,
    metrics_path: Path,
    metrics_display_path: Path | None = None,
    generated_at: str,
) -> str:
    corpus = dict(readiness.get("corpus") or {})
    dispatch = dict(readiness.get("dispatch") or {})
    classes = dict(readiness.get("execution_classes") or {})
    status = str(readiness.get("status") or "unknown")
    needed = int(readiness.get("needed_for_minimum", 0) or 0)
    min_dispatched = int(readiness.get("min_dispatched_for_first_slice", 0) or 0)
    verdict = (
        "Ready to promote the first canonical rev-4 slice."
        if status == "promotion_ready"
        else f"Not ready: needs {needed} more dispatched issue(s) to reach the {min_dispatched}-issue promotion floor."
    )

    lines = [
        "# H1-01 Rev-4 Promotion Readiness",
        "",
        f"Last updated: {generated_at}",
        "",
        "This is the operator-facing readiness surface for promoting the staged rev-4 benchmark corpus into the canonical B0 truth loop.",
        "",
        "## Verdict",
        "",
        f"- Status: `{status}`",
        f"- Decision: {verdict}",
        f"- Staging corpus: `{_repo_stable_path(corpus_path)}`",
        f"- Metrics source: `{_repo_stable_path(metrics_display_path or metrics_path)}`",
        f"- Promotion target: `{corpus.get('promotion_target') or 'n/a'}`",
        "",
        "## Corpus",
        "",
        f"- Corpus id: `{corpus.get('corpus_id') or 'n/a'}`",
        f"- Revision: `{corpus.get('revision') or 'n/a'}`",
        f"- Manifest status: `{corpus.get('status') or 'n/a'}`",
        f"- Recorded on: `{corpus.get('recorded_on') or 'n/a'}`",
        f"- Total staged issues: `{corpus.get('issue_count') or 0}`",
        "",
        "## Dispatch Evidence",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Dispatch floor for first canonical slice | {min_dispatched} |",
        f"| Staged issues with metrics evidence | {dispatch.get('dispatched_issue_count') or 0} |",
        f"| Staged issues still missing metrics evidence | {dispatch.get('missing_issue_count') or 0} |",
        f"| Additional dispatches needed | {needed} |",
        "",
        "## Next Dispatch Targets",
        "",
        _format_issue_ids(list(dispatch.get("recommended_next_issue_ids") or [])),
        "",
        "## Execution-Class Coverage",
        "",
        "| Execution class | Dispatched | Total | Missing |",
        "| --- | ---: | ---: | ---: |",
    ]
    for execution_class, values in classes.items():
        value_map = dict(values)
        lines.append(
            f"| `{execution_class}` | {int(value_map.get('dispatched', 0) or 0)} | "
            f"{int(value_map.get('total', 0) or 0)} | {int(value_map.get('missing', 0) or 0)} |"
        )

    lines.extend(
        [
            "",
            "## Dispatched Issues",
            "",
            _format_issue_ids(list(dispatch.get("dispatched_issue_ids") or [])),
            "",
            "## Missing Evidence",
            "",
            _format_issue_ids(list(dispatch.get("missing_issue_ids") or [])),
            "",
            "## Promotion Rule",
            "",
            "Promote only a first canonical rev-4 slice after at least 15 staged entries have dispatch evidence in `boss_metrics.jsonl`. Keep undispatched entries staged until they also accumulate evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-dispatched", type=int, default=DEFAULT_MIN_DISPATCHED)
    parser.add_argument("--generated-at", help=argparse.SUPPRESS)
    parser.add_argument(
        "--json", action="store_true", help="Emit readiness JSON instead of Markdown"
    )
    parser.add_argument(
        "--fail-not-ready",
        action="store_true",
        help="Exit non-zero when the promotion floor has not been met",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus_path = args.corpus.resolve()
    metrics_path = resolve_metrics_path(args.metrics)
    corpus = load_corpus(corpus_path)
    readiness = build_readiness(
        corpus=corpus,
        metrics_rows=load_metrics(metrics_path),
        min_dispatched=args.min_dispatched,
    )
    generated_at = _normalize_generated_at(args.generated_at)

    if args.json:
        print(json.dumps(readiness, indent=2, sort_keys=True))
    else:
        markdown = render_markdown(
            readiness=readiness,
            corpus_path=corpus_path,
            metrics_path=metrics_path,
            metrics_display_path=args.metrics,
            generated_at=generated_at,
        )
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(str(output_path))

    if args.fail_not_ready and readiness.get("status") != "promotion_ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

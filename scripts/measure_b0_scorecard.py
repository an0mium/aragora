#!/usr/bin/env python3
"""B0 Benchmark Scorecard: measure autonomous execution truth metrics.

Reads boss_metrics.jsonl and produces a no-rescue scorecard showing:
- Total ticks / unique issues attempted
- No-rescue success rate (deliverable_pr_created without human intervention)
- Terminal class distribution
- Failure class breakdown
- Per-category success rates (when outcome learner data exists)

This is the primary truth metric for the wedge-first roadmap (TW-01/TW-02).

Usage:
  python3 scripts/measure_b0_scorecard.py
  python3 scripts/measure_b0_scorecard.py --metrics .aragora/overnight/boss_metrics.jsonl
  python3 scripts/measure_b0_scorecard.py --json
  python3 scripts/measure_b0_scorecard.py --ci --threshold 0.5
  python3 scripts/measure_b0_scorecard.py --window 100
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from build_benchmark_truth_artifact import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_PUBLISH_DIR as DEFAULT_TRUTH_ARTIFACT_PUBLISH_DIR,
    build_benchmark_truth_artifact,
    load_corpus as load_benchmark_corpus,
    resolve_published_artifact_path as resolve_truth_artifact_path,
    write_artifact as write_truth_artifact,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_PATH = Path(".aragora/overnight/boss_metrics.jsonl")
DEFAULT_PUBLISH_DIR = REPO_ROOT / ".aragora" / "benchmark_scorecards"

# Terminal classes that count as autonomous success (no human rescue)
SUCCESS_CLASSES = frozenset(
    {
        "success_merged",
        "success_pr_created",
        "deliverable_pr_created",
    }
)

# Terminal classes that indicate the system tried and failed
FAILURE_CLASSES = frozenset(
    {
        "rescue_timeout",
        "rescue_worker_crash",
        "rescue_no_deliverable",
        "blocked_not_dispatch_bounded",
        "blocked_validation_target_missing",
        "blocked_sanitation_failed",
        "blocked_auth_failure",
        "blocked_no_runner",
    }
)


def _coerce_utc_datetime(value: str | None = None) -> dt.datetime:
    if value:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = dt.datetime.now(dt.UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC).replace(microsecond=0)


def normalize_generated_at(value: str | None = None) -> str:
    return _coerce_utc_datetime(value).isoformat().replace("+00:00", "Z")


def _repo_stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "benchmark-corpus"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return payload


def load_corpus_issue_numbers(path: Path) -> tuple[dict[str, Any], set[int]]:
    payload = load_benchmark_corpus(path)
    issue_numbers: set[int] = set()
    for item in payload.get("issues", []):
        if not isinstance(item, dict):
            continue
        issue_number = int(item.get("issue_id", 0) or 0)
        if issue_number > 0:
            issue_numbers.add(issue_number)
    return payload, issue_numbers


def load_truth_artifact_issue_numbers(path: Path) -> tuple[dict[str, Any], set[int]]:
    payload = _load_json(path)
    issue_numbers: set[int] = set()
    for item in payload.get("issues", []):
        if not isinstance(item, dict):
            continue
        issue_number = int(item.get("issue_number", 0) or 0)
        if issue_number > 0:
            issue_numbers.add(issue_number)
    return payload, issue_numbers


def filter_rows_to_issue_numbers(
    rows: list[dict[str, Any]],
    issue_numbers: set[int],
) -> list[dict[str, Any]]:
    if not issue_numbers:
        return []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        issue_number = row.get("issue_number")
        if isinstance(issue_number, int) and issue_number in issue_numbers:
            filtered.append(row)
    return filtered


def build_scorecard_corpus_metadata(
    *,
    corpus_payload: dict[str, Any],
    issue_numbers: set[int],
    rows: list[dict[str, Any]],
    source_path: Path,
) -> dict[str, Any]:
    attempted_issue_numbers = {
        row["issue_number"]
        for row in rows
        if isinstance(row.get("issue_number"), int) and row["issue_number"] in issue_numbers
    }
    missing_issue_numbers = sorted(issue_numbers - attempted_issue_numbers)
    return {
        "corpus": {
            "path": _repo_stable_path(source_path),
            "corpus_id": str(corpus_payload.get("corpus_id") or "").strip(),
            "revision": int(corpus_payload.get("revision", 0) or 0),
            "recorded_on": str(corpus_payload.get("recorded_on") or "").strip() or None,
            "success_contract": str(corpus_payload.get("success_contract") or "").strip() or None,
            "issue_count": len(issue_numbers),
        },
        "coverage": {
            "attempted_issue_count": len(attempted_issue_numbers),
            "missing_issue_count": len(missing_issue_numbers),
            "missing_issue_numbers": missing_issue_numbers,
            "is_complete": not missing_issue_numbers,
            "status": "complete" if not missing_issue_numbers else "incomplete",
        },
    }


def auto_publish_truth_artifact(
    *,
    repo: str,
    metrics_path: Path,
    corpus_path: Path,
    truth_publish_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    artifact = build_benchmark_truth_artifact(
        repo=repo,
        metrics_file=metrics_path,
        corpus_path=corpus_path,
    )
    published_path = write_truth_artifact(
        resolve_truth_artifact_path(
            publish_dir=truth_publish_dir,
            artifact=artifact,
        ),
        artifact,
    )
    return published_path, artifact


def _metric_value(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _comparison_delta(
    current: float | None, previous: float | None, *, digits: int = 4
) -> float | None:
    if current is None or previous is None:
        return None
    return round(current - previous, digits)


def resolve_published_scorecard_path(
    *,
    publish_dir: Path,
    published_scorecard: dict[str, Any],
) -> Path:
    corpus = dict(published_scorecard.get("corpus") or {})
    timestamp = _coerce_utc_datetime(str(published_scorecard.get("generated_at") or None)).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    corpus_id = _slugify(str(corpus.get("corpus_id") or "benchmark-corpus"))
    revision = int(corpus.get("revision", 0) or 0)
    return publish_dir / corpus_id / f"rev-{revision}" / f"scorecard-{timestamp}.json"


def resolve_available_published_scorecard_path(
    *,
    publish_dir: Path,
    published_scorecard: dict[str, Any],
) -> Path:
    base_path = resolve_published_scorecard_path(
        publish_dir=publish_dir,
        published_scorecard=published_scorecard,
    )
    if not base_path.exists():
        return base_path
    for index in range(2, 1000):
        candidate = base_path.with_name(f"{base_path.stem}-{index}{base_path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to reserve unique scorecard path under {base_path.parent}")


def _previous_published_scorecard_path(
    *,
    publish_dir: Path,
    corpus_id: str,
    revision: int,
) -> Path | None:
    target_dir = publish_dir / _slugify(corpus_id) / f"rev-{revision}"
    candidates = sorted(target_dir.glob("scorecard-*.json"))
    return candidates[-1] if candidates else None


def build_published_scorecard(
    *,
    scorecard: dict[str, Any],
    metrics_path: Path,
    truth_artifact_path: Path,
    publish_dir: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    truth_artifact = _load_json(truth_artifact_path)
    corpus = dict(truth_artifact.get("corpus") or {})
    if not corpus:
        raise ValueError(f"Truth artifact at {truth_artifact_path} is missing corpus metadata")

    published = {
        "generated_at": normalize_generated_at(generated_at),
        "metrics_file": _repo_stable_path(metrics_path),
        "truth_artifact_path": _repo_stable_path(truth_artifact_path),
        "truth_artifact_generated_at": str(truth_artifact.get("generated_at") or "").strip()
        or None,
        "corpus": {
            "path": str(corpus.get("path") or "").strip() or None,
            "corpus_id": str(corpus.get("corpus_id") or "").strip(),
            "revision": int(corpus.get("revision", 0) or 0),
            "recorded_on": str(corpus.get("recorded_on") or "").strip() or None,
            "success_contract": str(corpus.get("success_contract") or "").strip() or None,
            "issue_count": int(corpus.get("issue_count", 0) or 0),
        },
        "truth_metrics": dict(truth_artifact.get("primary_metrics") or {}),
        "coverage": dict(truth_artifact.get("coverage") or {}),
        "failure_class_distribution": dict(truth_artifact.get("failure_class_distribution") or {}),
        "rescue_counts_by_type": dict(truth_artifact.get("rescue_counts_by_type") or {}),
        "proxy_metrics": dict(scorecard),
    }

    previous_path = _previous_published_scorecard_path(
        publish_dir=publish_dir,
        corpus_id=published["corpus"]["corpus_id"],
        revision=int(published["corpus"]["revision"] or 0),
    )
    if previous_path is not None:
        previous = _load_json(previous_path)
        published["previous_artifact"] = {
            "path": _repo_stable_path(previous_path),
            "generated_at": str(previous.get("generated_at") or "").strip() or None,
        }
        published["deltas"] = {
            "truth_success_rate": _comparison_delta(
                _metric_value(published, "truth_metrics", "truth_success_rate"),
                _metric_value(previous, "truth_metrics", "truth_success_rate"),
            ),
            "no_rescue_truth_success_rate": _comparison_delta(
                _metric_value(published, "truth_metrics", "no_rescue_truth_success_rate"),
                _metric_value(previous, "truth_metrics", "no_rescue_truth_success_rate"),
            ),
            "merged_only_rate": _comparison_delta(
                _metric_value(published, "truth_metrics", "merged_only_rate"),
                _metric_value(previous, "truth_metrics", "merged_only_rate"),
            ),
            "proxy_no_rescue_success_rate": _comparison_delta(
                _metric_value(published, "proxy_metrics", "no_rescue_success_rate"),
                _metric_value(previous, "proxy_metrics", "no_rescue_success_rate"),
            ),
            "unique_issues_attempted": _comparison_delta(
                _metric_value(published, "proxy_metrics", "unique_issues_attempted"),
                _metric_value(previous, "proxy_metrics", "unique_issues_attempted"),
                digits=0,
            ),
        }
    return published


def write_artifact(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_metrics(path: Path, window: int | None = None) -> list[dict[str, Any]]:
    """Load boss metrics JSONL, optionally limiting to last N rows."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    if window and window > 0:
        return rows[-window:]
    return rows


def compute_scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the B0 benchmark scorecard from metrics rows."""
    if not rows:
        return {"status": "no_data", "total_ticks": 0}

    terminal_classes: Counter[str] = Counter()
    issues_attempted: set[int] = set()
    issues_succeeded: set[int] = set()
    issues_failed: set[int] = set()
    elapsed_times: list[float] = []

    for row in rows:
        tc = str(row.get("terminal_class", "")).strip()
        if tc:
            terminal_classes[tc] += 1

        issue_num = row.get("issue_number")
        if isinstance(issue_num, int) and issue_num > 0:
            issues_attempted.add(issue_num)
            if tc in SUCCESS_CLASSES:
                issues_succeeded.add(issue_num)
            elif tc in FAILURE_CLASSES:
                issues_failed.add(issue_num)

        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and elapsed > 0:
            elapsed_times.append(float(elapsed))

    total_ticks = len(rows)
    success_ticks = sum(terminal_classes[tc] for tc in SUCCESS_CLASSES)
    failure_ticks = sum(terminal_classes[tc] for tc in FAILURE_CLASSES)

    return {
        "status": "active",
        "total_ticks": total_ticks,
        "unique_issues_attempted": len(issues_attempted),
        "unique_issues_succeeded": len(issues_succeeded),
        "unique_issues_failed": len(issues_failed),
        "no_rescue_success_rate": round(len(issues_succeeded) / len(issues_attempted), 3)
        if issues_attempted
        else 0.0,
        "tick_success_rate": round(success_ticks / total_ticks, 3) if total_ticks else 0.0,
        "terminal_class_distribution": dict(terminal_classes.most_common()),
        "success_classes": {
            tc: terminal_classes[tc] for tc in sorted(SUCCESS_CLASSES) if terminal_classes[tc]
        },
        "failure_classes": {
            tc: terminal_classes[tc] for tc in sorted(FAILURE_CLASSES) if terminal_classes[tc]
        },
        "median_elapsed_seconds": round(sorted(elapsed_times)[len(elapsed_times) // 2], 1)
        if elapsed_times
        else 0.0,
        "mean_elapsed_seconds": round(sum(elapsed_times) / len(elapsed_times), 1)
        if elapsed_times
        else 0.0,
    }


def print_scorecard(scorecard: dict[str, Any]) -> None:
    """Print human-readable scorecard."""
    if scorecard.get("status") == "no_data":
        print("No boss metrics data found.")
        return

    print("=" * 60)
    print("  B0 BENCHMARK SCORECARD")
    print("=" * 60)
    print(f"  Total ticks:              {scorecard['total_ticks']}")
    print(f"  Unique issues attempted:  {scorecard['unique_issues_attempted']}")
    print(f"  Unique issues succeeded:  {scorecard['unique_issues_succeeded']}")
    print(f"  Unique issues failed:     {scorecard['unique_issues_failed']}")
    print(f"  No-rescue success rate:   {scorecard['no_rescue_success_rate']:.1%}")
    print(f"  Per-tick success rate:    {scorecard['tick_success_rate']:.1%}")
    print(f"  Median elapsed time:      {scorecard['median_elapsed_seconds']:.0f}s")
    print(f"  Mean elapsed time:        {scorecard['mean_elapsed_seconds']:.0f}s")
    print()
    print("  Terminal class distribution:")
    for tc, count in scorecard.get("terminal_class_distribution", {}).items():
        marker = "+" if tc in SUCCESS_CLASSES else "-" if tc in FAILURE_CLASSES else " "
        print(f"    {marker} {tc}: {count}")
    print("=" * 60)


def render_ci_summary(scorecard: dict[str, Any], *, threshold: float) -> str:
    success_rate = float(scorecard.get("no_rescue_success_rate", 0.0) or 0.0)
    coverage_status = str((scorecard.get("coverage") or {}).get("status") or "n/a")
    status = (
        "pass" if success_rate >= threshold and scorecard.get("status") != "no_data" else "fail"
    )
    return " ".join(
        [
            f"status={status}",
            f"scorecard_status={scorecard.get('status', 'unknown')}",
            f"success_rate={success_rate:.3f}",
            f"threshold={threshold:.3f}",
            f"coverage_status={coverage_status}",
            f"total_ticks={int(scorecard.get('total_ticks', 0) or 0)}",
            f"unique_issues_attempted={int(scorecard.get('unique_issues_attempted', 0) or 0)}",
            f"unique_issues_succeeded={int(scorecard.get('unique_issues_succeeded', 0) or 0)}",
            f"unique_issues_failed={int(scorecard.get('unique_issues_failed', 0) or 0)}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B0 Benchmark Scorecard")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to boss_metrics.jsonl",
    )
    parser.add_argument("--window", type=int, default=None, help="Last N ticks only")
    parser.add_argument(
        "--truth-artifact",
        type=Path,
        default=None,
        help="Optional benchmark truth artifact JSON used for published recurring scorecards",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=f"Optional fixed benchmark corpus manifest (default in auto-publish mode: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument(
        "--repo",
        default="synaptent/aragora",
        help="GitHub repo owner/name used when auto-building a truth artifact",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--json", action="store_true", help="JSON output")
    output_group.add_argument(
        "--ci",
        action="store_true",
        help="CI output: one-line summary and threshold-based exit status",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum no-rescue success rate required in --ci mode (default: 0.5)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Write a timestamped scorecard artifact under the repo-stable publish path",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=None,
        help=f"Optional publish root override (default: {DEFAULT_PUBLISH_DIR})",
    )
    parser.add_argument(
        "--truth-publish-dir",
        type=Path,
        default=None,
        help=(
            "Optional truth-artifact publish root when --publish auto-builds from --corpus "
            f"(default: {DEFAULT_TRUTH_ARTIFACT_PUBLISH_DIR})"
        ),
    )
    parser.add_argument(
        "--fail-incomplete",
        action="store_true",
        help="Exit non-zero when the selected corpus does not appear completely in the metrics window",
    )
    args = parser.parse_args(argv)

    metrics_path = args.metrics.resolve()
    rows = load_metrics(metrics_path, args.window)
    publish_dir: Path | None = None
    if args.publish_dir is not None:
        publish_dir = args.publish_dir.resolve()
    elif args.publish:
        publish_dir = DEFAULT_PUBLISH_DIR
    truth_artifact_path = args.truth_artifact.resolve() if args.truth_artifact else None
    corpus_path = args.corpus.resolve() if args.corpus else None
    if truth_artifact_path is None and publish_dir is not None and corpus_path is None:
        corpus_path = DEFAULT_CORPUS_PATH

    corpus_issue_numbers: set[int] | None = None
    corpus_metadata_payload: dict[str, Any] | None = None
    corpus_metadata_source_path: Path | None = None
    if corpus_path is not None:
        if not corpus_path.exists():
            raise SystemExit(f"corpus file not found: {corpus_path}")
        corpus_metadata_payload, corpus_issue_numbers = load_corpus_issue_numbers(corpus_path)
        corpus_metadata_source_path = corpus_path
    elif truth_artifact_path is not None:
        if not truth_artifact_path.exists():
            raise SystemExit(f"truth artifact not found: {truth_artifact_path}")
        truth_payload, truth_issue_numbers = load_truth_artifact_issue_numbers(truth_artifact_path)
        truth_corpus = dict(truth_payload.get("corpus") or {})
        if truth_corpus:
            corpus_metadata_payload = truth_corpus
            if truth_issue_numbers:
                corpus_issue_numbers = truth_issue_numbers
                truth_artifact_recorded_path = str(truth_corpus.get("path") or "").strip()
                corpus_metadata_source_path = (
                    REPO_ROOT / truth_artifact_recorded_path
                    if truth_artifact_recorded_path
                    else truth_artifact_path
                )

    selected_rows = (
        filter_rows_to_issue_numbers(rows, corpus_issue_numbers)
        if corpus_issue_numbers is not None
        else rows
    )
    scorecard = compute_scorecard(selected_rows)
    if corpus_metadata_payload is not None and corpus_issue_numbers is not None:
        source_path = corpus_metadata_source_path or truth_artifact_path or metrics_path
        scorecard.update(
            build_scorecard_corpus_metadata(
                corpus_payload=corpus_metadata_payload,
                issue_numbers=corpus_issue_numbers,
                rows=selected_rows,
                source_path=source_path,
            )
        )

    published_scorecard: dict[str, Any] | None = None
    published_path: Path | None = None
    if publish_dir is not None:
        if truth_artifact_path is None:
            if corpus_path is None:
                raise SystemExit("publish mode requires --truth-artifact PATH or --corpus PATH")
            truth_publish_dir = (
                args.truth_publish_dir.resolve()
                if args.truth_publish_dir is not None
                else DEFAULT_TRUTH_ARTIFACT_PUBLISH_DIR
            )
            truth_artifact_path, truth_payload = auto_publish_truth_artifact(
                repo=str(args.repo),
                metrics_path=metrics_path,
                corpus_path=corpus_path,
                truth_publish_dir=truth_publish_dir,
            )
            if corpus_metadata_payload is None or corpus_issue_numbers is None:
                truth_corpus = dict(truth_payload.get("corpus") or {})
                corpus_metadata_payload = truth_corpus
                corpus_issue_numbers = {
                    int(item.get("issue_number", 0) or 0)
                    for item in truth_payload.get("issues", [])
                    if isinstance(item, dict) and int(item.get("issue_number", 0) or 0) > 0
                }
                corpus_metadata_source_path = corpus_path
        published_scorecard = build_published_scorecard(
            scorecard=scorecard,
            metrics_path=metrics_path,
            truth_artifact_path=truth_artifact_path,
            publish_dir=publish_dir,
        )
        published_path = write_artifact(
            resolve_available_published_scorecard_path(
                publish_dir=publish_dir,
                published_scorecard=published_scorecard,
            ),
            published_scorecard,
        )
        if args.json or args.ci:
            print(str(published_path), file=sys.stderr)
        else:
            print(str(published_path))

    if args.json:
        print(json.dumps(published_scorecard or scorecard, indent=2, sort_keys=True))
    elif args.ci:
        print(render_ci_summary(scorecard, threshold=args.threshold))
        is_complete = bool((scorecard.get("coverage") or {}).get("is_complete", True))
        return (
            0
            if scorecard.get("status") != "no_data"
            and scorecard.get("no_rescue_success_rate", 0.0) >= args.threshold
            and (not args.fail_incomplete or is_complete)
            else 1
        )
    elif published_path is None:
        print_scorecard(scorecard)

    if args.fail_incomplete and (scorecard.get("coverage") or {}).get("is_complete") is False:
        missing_issue_numbers = list(
            (scorecard.get("coverage") or {}).get("missing_issue_numbers") or []
        )
        missing_suffix = ", ".join(str(item) for item in missing_issue_numbers) or "unknown"
        print(
            f"incomplete corpus coverage: missing issue numbers {missing_suffix}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

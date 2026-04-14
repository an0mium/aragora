#!/usr/bin/env python3
"""Rotate benchmark metrics and run the open remainder of the fixed corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.boss_loop_outcome import append_iteration_metrics
from build_benchmark_truth_artifact import DEFAULT_CORPUS_PATH, load_corpus
from rotate_boss_metrics import DEFAULT_METRICS_PATH, rotate_metrics_file

DEFAULT_REPO = "synaptent/aragora"
DEFAULT_OUTCOME_LEARNER_WINDOW = 500


def corpus_issue_numbers(corpus: dict[str, Any]) -> list[int]:
    numbers = [
        int(item.get("issue_id", 0) or 0)
        for item in list(corpus.get("issues") or [])
        if isinstance(item, dict) and int(item.get("issue_id", 0) or 0) > 0
    ]
    return sorted(dict.fromkeys(numbers))


def corpus_issue_titles(corpus: dict[str, Any]) -> dict[int, str]:
    titles: dict[int, str] = {}
    for item in list(corpus.get("issues") or []):
        if not isinstance(item, dict):
            continue
        issue_number = int(item.get("issue_id", 0) or 0)
        if issue_number <= 0:
            continue
        titles[issue_number] = str(item.get("title") or "").strip()
    return titles


def _run_json_object(
    cmd: list[str],
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    proc = runner(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "command failed")
    payload = json.loads(proc.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("command did not return a JSON object")
    return payload


def filter_open_issue_numbers(
    repo: str,
    issue_numbers: list[int],
    *,
    runner: Any = subprocess.run,
) -> list[int]:
    open_numbers: list[int] = []
    for issue_number in issue_numbers:
        payload = _run_json_object(
            [
                "gh",
                "issue",
                "view",
                str(issue_number),
                "--repo",
                repo,
                "--json",
                "number,state",
            ],
            runner=runner,
        )
        if str(payload.get("state") or "").strip().upper() == "OPEN":
            open_numbers.append(issue_number)
    return open_numbers


def append_closed_issue_rows(
    *,
    metrics_file: Path,
    closed_issue_numbers: list[int],
    issue_titles: dict[int, str],
    dry_run: bool = False,
) -> list[int]:
    if dry_run:
        return list(closed_issue_numbers)

    appended: list[int] = []
    for iteration, issue_number in enumerate(closed_issue_numbers, start=1):
        title = issue_titles.get(issue_number, "")
        append_iteration_metrics(
            metrics_jsonl_path=str(metrics_file),
            outcome_learner_window=DEFAULT_OUTCOME_LEARNER_WINDOW,
            deferred_queue_depth=0,
            iteration=iteration,
            issue_number=issue_number,
            worker_result={
                "status": "completed",
                "outcome": "issue_already_resolved",
                "issue_title": title,
                "receipt_metadata": {"issue_title": title},
            },
            elapsed_seconds=0.0,
            files_changed=0,
            tests_run=0,
            tests_passed=0,
        )
        appended.append(issue_number)
    return appended


def build_boss_loop_command(
    *,
    repo: str,
    issue_numbers: list[int],
    max_ticks: int | None = None,
    interval_seconds: float = 30.0,
    max_consecutive_failures: int = 5,
    autonomy: str = "fire_and_forget",
    max_hours: float = 10.0,
) -> list[str]:
    if not issue_numbers:
        raise ValueError("issue_numbers must not be empty")
    effective_ticks = max_ticks if max_ticks is not None else max(len(issue_numbers) * 2, 1)
    return [
        sys.executable,
        "-m",
        "aragora.cli.main",
        "swarm",
        "boss-loop",
        "--boss-repo",
        repo,
        "--boss-issue-list",
        ",".join(str(item) for item in issue_numbers),
        "--max-ticks",
        str(effective_ticks),
        "--interval",
        str(interval_seconds),
        "--max-consecutive-failures",
        str(max(max_consecutive_failures, len(issue_numbers))),
        "--autonomy",
        autonomy,
        "--max-hours",
        str(max_hours),
    ]


def recorded_issue_numbers(metrics_file: Path) -> list[int]:
    if not metrics_file.exists():
        return []
    recorded: set[int] = set()
    for line in metrics_file.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        issue_number = payload.get("issue_number")
        if isinstance(issue_number, int) and issue_number > 0:
            recorded.add(issue_number)
    return sorted(recorded)


def run_recurrence(
    *,
    corpus_path: Path,
    repo: str,
    metrics_file: Path,
    max_ticks: int | None = None,
    interval_seconds: float = 30.0,
    max_consecutive_failures: int = 5,
    autonomy: str = "fire_and_forget",
    max_hours: float = 10.0,
    dry_run: bool = False,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    corpus = load_corpus(corpus_path)
    issue_numbers = corpus_issue_numbers(corpus)
    issue_titles = corpus_issue_titles(corpus)
    open_issue_numbers = filter_open_issue_numbers(repo, issue_numbers, runner=runner)
    closed_issue_numbers = sorted(set(issue_numbers) - set(open_issue_numbers))
    rotate_summary = rotate_metrics_file(metrics_file, dry_run=dry_run)
    synthetic_resolved_issue_numbers = append_closed_issue_rows(
        metrics_file=metrics_file,
        closed_issue_numbers=closed_issue_numbers,
        issue_titles=issue_titles,
        dry_run=dry_run,
    )
    command = (
        build_boss_loop_command(
            repo=repo,
            issue_numbers=open_issue_numbers,
            max_ticks=max_ticks,
            interval_seconds=interval_seconds,
            max_consecutive_failures=max_consecutive_failures,
            autonomy=autonomy,
            max_hours=max_hours,
        )
        if open_issue_numbers
        else None
    )
    boss_loop_exit_code: int | None = None
    if command is not None and not dry_run:
        proc = runner(command, check=False, cwd=str(REPO_ROOT))
        boss_loop_exit_code = int(proc.returncode)

    recorded_numbers = issue_numbers if dry_run else recorded_issue_numbers(metrics_file)
    missing_issue_numbers = sorted(set(issue_numbers) - set(recorded_numbers))
    if not dry_run and not boss_loop_exit_code and missing_issue_numbers:
        raise RuntimeError(
            "incomplete recurring benchmark corpus metrics: missing issue numbers "
            + ", ".join(str(item) for item in missing_issue_numbers)
        )

    return {
        "repo": repo,
        "corpus_path": str(corpus_path),
        "metrics_file": str(metrics_file),
        "issue_numbers": issue_numbers,
        "open_issue_numbers": open_issue_numbers,
        "closed_issue_numbers": closed_issue_numbers,
        "synthetic_resolved_issue_numbers": synthetic_resolved_issue_numbers,
        "recorded_issue_numbers": recorded_numbers,
        "missing_issue_numbers": missing_issue_numbers,
        "rotated_metrics": rotate_summary,
        "boss_loop_command": command,
        "boss_loop_exit_code": boss_loop_exit_code,
        "dry_run": dry_run,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Benchmark corpus manifest (default: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Metrics file rotated and fed into publish (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--autonomy", default="fire_and_forget")
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_recurrence(
        corpus_path=args.corpus.resolve(),
        repo=str(args.repo),
        metrics_file=args.metrics_file.resolve(),
        max_ticks=args.max_ticks,
        interval_seconds=args.interval_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
        autonomy=str(args.autonomy),
        max_hours=float(args.max_hours),
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Recurring benchmark corpus feed: "
            f"{len(summary['open_issue_numbers'])} open issue(s), "
            f"{len(summary['closed_issue_numbers'])} closed issue(s)."
        )
        if summary["boss_loop_command"] is None:
            print(
                "No open corpus issues remain; rotated metrics file and skipped boss-loop dispatch."
            )
        elif args.dry_run:
            print("Would run:", " ".join(summary["boss_loop_command"]))
        elif summary["boss_loop_exit_code"] is not None:
            print(f"Boss loop exit code: {summary['boss_loop_exit_code']}")
    return int(summary["boss_loop_exit_code"] or 0)


if __name__ == "__main__":
    raise SystemExit(main())
